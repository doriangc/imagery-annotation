"""Predict the BGT classes of a central target square from a larger square window around it.

Two inputs:
- imagery (default): the aerial photo crop, on the same pixel grid as the labels.
- labels: the class raster itself with the target square hidden, so the model has to infer it from context.
  This was the first step for setting up the code, and stays as a regression test with a copy-left baseline.

The model predicts every pixel, but only the target square is trained on: its pixels all have at least
`(size - target) / 2` px of context on every side. A large target (e.g. 384 in a 512 window) supervises most of
the window, at the cost of less context near its edge. Evaluation windows are scored on their central SCORE px
whatever the target, so runs with different targets stay comparable; --full-eval also scores every pixel of
the validation strips and test areas, tiled with the model's target squares.
With --ema, evaluation and the saved checkpoint use an exponential moving average of the weights.
Training windows get a random 90° rotation and flip, and with --lighting random lighting changes and shadows;
validation windows stay as they are.
Runs on a GPU when one is available, with windows sampled by DataLoader worker processes.

Areas (see areas.py) are read from data/<area>_classes_8cm.* and data/<area>_aerial_2026.npy. Training areas
(--areas) hold out their southern strip for validation; test areas (--test-areas) are held out entirely.

With --context-scale N, each window also gets a context view: the same number of pixels around the same centre,
downsampled N times (from data/<area>_aerial_2026_ctxN.npy, built on first use), stacked as 3 extra channels.
Both views share a centre, so rotations and flips keep them consistent.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

VAL_FRACTION = 0.15  # the southern strip of the raster is held out
IGNORE = 255  # label of pixels left out of the loss and scores (rasterize.IGNORE)
MAX_SIZE = 512  # largest supported window; centres keep this margin, so every size scores the same target squares
SCORE = 128  # side of the central square evaluation windows are scored on (the target, if smaller)
DEFAULTS = {"imagery": (256, 128), "labels": (160, 32)}  # (size, target) per input
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # so pretrained encoders can drop in later
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def context_path(imagery_path: str, scale: int) -> str:
    return imagery_path.removesuffix(".npy") + f"_ctx{scale}.npy"


def build_context(imagery_path: str, scale: int, rows_per_chunk: int = 2048) -> str:
    """Write a scale x scale block-mean downsampled copy of an imagery raster, if it doesn't exist yet."""
    path = context_path(imagery_path, scale)
    if os.path.exists(path):
        return path
    imagery = np.load(imagery_path, mmap_mode="r")
    height, width = imagery.shape[0] // scale, imagery.shape[1] // scale
    tmp = path.removesuffix(".npy") + ".tmp.npy"  # written under another name, so an interrupted build isn't used
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(height, width, 3))
    step = rows_per_chunk // scale
    for r in range(0, height, step):
        n = min(step, height - r)
        block = np.asarray(imagery[r * scale:(r + n) * scale, :width * scale], np.float32)
        out[r:r + n] = (block.reshape(n, scale, width, scale, 3).mean(axis=(1, 3)) + 0.5).astype(np.uint8)
    out.flush()
    del out
    os.replace(tmp, path)
    return path


def crop(raster: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    """raster[row:row + size, col:col + size], zero-padded where it extends past the edges."""
    out = np.zeros((size, size) + raster.shape[2:], raster.dtype)
    r0, c0 = max(row, 0), max(col, 0)
    r1, c1 = min(row + size, raster.shape[0]), min(col + size, raster.shape[1])
    if r0 < r1 and c0 < c1:
        out[r0 - row:r1 - row, c0 - col:c1 - col] = raster[r0:r1, c0:c1]
    return out


def context_window(context: np.ndarray, scale: int, row: int, col: int, size: int) -> np.ndarray:
    """The context view for the full-resolution window with top-left (row, col): size px around its centre."""
    return crop(context, (row + size // 2) // scale - size // 2, (col + size // 2) // scale - size // 2, size)


class Windows:
    """Random square windows (labels, and optionally the matching imagery). split is "train" (north),
    "val" (the southern VAL_FRACTION strip, so train and val never overlap) or "all" (a held-out test area).

    Window centres are drawn from a range that doesn't depend on `size`, so with the same seed, runs with
    different window sizes are scored on exactly the same target squares."""

    def __init__(self, labels_path: str, imagery_path: str | None, split: str, size: int, seed: int = 0,
                 context_scale: int = 0):
        assert size <= MAX_SIZE, f"windows up to {MAX_SIZE} px are supported"
        self.labels = np.load(labels_path, mmap_mode="r")
        self.imagery = np.load(imagery_path, mmap_mode="r") if imagery_path else None
        self.context_scale = context_scale
        self.context = np.load(context_path(imagery_path, context_scale), mmap_mode="r") if context_scale else None
        if self.imagery is not None:
            assert self.imagery.shape[:2] == self.labels.shape, "imagery and labels must share the pixel grid"
        self.size = size
        height, width = self.labels.shape
        val_top = int(height * (1 - VAL_FRACTION))
        half = MAX_SIZE // 2
        self.row_range = {"train": (half, val_top - half), "val": (val_top + half, height - half),
                          "all": (half, height - half)}[split]
        self.col_range = (half, width - half)
        self.rng = np.random.default_rng(seed)

    @property
    def num_centres(self) -> int:
        return (self.row_range[1] - self.row_range[0]) * (self.col_range[1] - self.col_range[0])

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """(size, size) labels and (size, size, 3) uint8 imagery, or 6 channels with the context view after
        the photo; the imagery is empty without an imagery path."""
        row = self.rng.integers(*self.row_range) - self.size // 2
        col = self.rng.integers(*self.col_range) - self.size // 2
        labels = np.array(self.labels[row:row + self.size, col:col + self.size])
        if self.imagery is None:
            return labels, np.zeros((0, 0, 3), np.uint8)
        image = np.array(self.imagery[row:row + self.size, col:col + self.size])
        if self.context is not None:
            image = np.concatenate([image, context_window(self.context, self.context_scale, row, col, self.size)], 2)
        return labels, image

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n samples stacked, as the DataLoader would collate them."""
        labels, images = zip(*(self.sample() for _ in range(n)))
        return torch.from_numpy(np.stack(labels)), torch.from_numpy(np.stack(images))


def augment(labels: np.ndarray, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random one of the 8 rotations/flips, applied identically to a window's labels and imagery."""
    k, flip = rng.integers(4), rng.random() < 0.5
    labels, image = np.rot90(labels, k), np.rot90(image, k)  # rot90 turns the first two axes, so RGB is kept
    if flip:
        labels, image = labels[:, ::-1], image[:, ::-1]
    return np.ascontiguousarray(labels), np.ascontiguousarray(image)


def photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random lighting changes of a uint8 RGB window: brightness, contrast, gamma, saturation, a slight colour
    cast, and half the time a synthetic shadow. Teaches the model not to rely on exact grey values, which vary
    with shadows, the sun and the photo flight."""
    x = image.astype(np.float32) / 255
    if rng.random() < 0.5:  # a soft-edged random polygon, darker and slightly blue, like a cast shadow
        size = image.shape[0]
        corners = rng.uniform(-0.25, 1.25, (int(rng.integers(3, 7)), 2)) * size
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in corners], fill=255)
        mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(rng.uniform(1, 6))), np.float32)[..., None] / 255
        shade = rng.uniform(0.35, 0.75) * np.array([0.95, 1.0, 1.1], np.float32)
        x = x * (1 - mask * (1 - shade))
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.7, 1.3)  # saturation
    x = (x - x.mean()) * rng.uniform(0.75, 1.25) + x.mean()  # contrast
    x = x * rng.uniform(0.75, 1.25) * rng.uniform(0.95, 1.05, 3).astype(np.float32)  # brightness, colour cast
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)  # gamma
    return (x * 255 + 0.5).astype(np.uint8)


class TrainStream(IterableDataset):
    """Endless augmented training windows from one or more areas, each picked in proportion to its training
    area so every square metre is equally likely. Each DataLoader worker samples with its own seed."""

    def __init__(self, sources: list[tuple[str, str | None]], size: int, augment: bool, seed: int = 0,
                 lighting: bool = False, context_scale: int = 0):
        self.sources = sources  # (labels_path, imagery_path) per area
        self.size = size
        self.context_scale = context_scale
        self.augment = augment
        self.lighting = lighting
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        seed = 10_000 * self.seed + (info.id if info else 0)  # seeds 0 and 1 share no worker streams
        areas = [Windows(labels, imagery, "train", self.size, seed=seed + 100 * i, context_scale=self.context_scale)
                 for i, (labels, imagery) in enumerate(self.sources)]
        p = np.array([a.num_centres for a in areas], dtype=float)
        p /= p.sum()
        rng = np.random.default_rng(seed + 1000)
        while True:
            labels, image = areas[rng.choice(len(areas), p=p)].sample()
            if self.augment:
                labels, image = augment(labels, image, rng)
            if self.lighting and image.size:  # the same lighting for the photo and its context view
                state = int(rng.integers(2 ** 63))
                image = np.concatenate([photometric(image[..., i:i + 3], np.random.default_rng(state))
                                        for i in range(0, image.shape[2], 3)], 2)
            yield labels, image


def target_slice(size: int, target: int) -> slice:
    """The central `target` px of a `size` px window, along one axis."""
    start = (size - target) // 2
    return slice(start, start + target)


def encode_labels(labels: torch.Tensor, num_classes: int, target: int) -> torch.Tensor:
    """One-hot input with the target square blanked plus a mask channel."""
    t = target_slice(labels.shape[-1], target)
    x = F.one_hot(labels.masked_fill(labels == IGNORE, 0), num_classes).permute(0, 3, 1, 2).float()
    x[:, :, t, t] = 0
    mask = torch.zeros_like(x[:, :1])
    mask[:, :, t, t] = 1
    return torch.cat([x, mask], dim=1)


def encode_imagery(images: torch.Tensor) -> torch.Tensor:
    """(n, size, size, 3k) uint8 (k RGB views) -> ImageNet-normalised (n, 3k, size, size) float."""
    views = images.shape[-1] // 3
    mean, std = IMAGENET_MEAN.to(images.device).repeat(1, views, 1, 1), IMAGENET_STD.to(images.device).repeat(1, views, 1, 1)
    return (images.permute(0, 3, 1, 2).float() / 255 - mean) / std


def encode(labels, images, num_classes, target, input_kind, device, memory_format=torch.contiguous_format):
    """Model input and the target square's labels, on `device`. Encoding happens there, so only uint8 is copied."""
    labels, images = labels.to(device, non_blocking=True).long(), images.to(device, non_blocking=True)
    t = target_slice(labels.shape[-1], target)
    x = encode_labels(labels, num_classes, target) if input_kind == "labels" else encode_imagery(images)
    return x.contiguous(memory_format=memory_format), labels[:, t, t]


class PatchNet(nn.Module):
    """Dilated conv encoder at full resolution and a per-pixel head, cropped to the target square."""

    def __init__(self, in_channels: int, num_classes: int, target: int, width: int = 32):
        super().__init__()
        self.target = target
        layers, channels = [], in_channels
        for dilation in (1, 2, 4, 8, 16, 32):  # receptive field ~±63 px
            layers += [nn.Conv2d(channels, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU()]
            channels = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = target_slice(x.shape[-1], self.target)
        return self.head(self.encoder(x)[:, :, t, t])


class UNet(nn.Module):
    """U-Net with a timm/torchvision encoder (ImageNet-pretrained or not), cropped to the target square.

    With context_scale, the input has 3 more channels: a context view `context_scale` times coarser around the
    same centre. A second encoder reads it; its two deepest feature maps are resampled to the part covering the
    window and added to the U-Net's two deepest levels through 1x1 convolutions. Those start at zero, so
    training starts from the plain U-Net."""

    FUSED_LEVELS = 2

    def __init__(self, in_channels: int, num_classes: int, target: int, encoder: str, weights: str | None,
                 context_scale: int = 0):
        super().__init__()
        import segmentation_models_pytorch as smp  # only needed for this model
        self.target = target
        self.in_channels = in_channels
        self.context_scale = context_scale
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=in_channels, classes=num_classes)
        if context_scale:
            self.context_encoder = smp.encoders.get_encoder(encoder, in_channels=in_channels, weights=weights)
            channels = self.net.encoder.out_channels[-self.FUSED_LEVELS:]
            self.fuse = nn.ModuleList(nn.Conv2d(c_ctx, c, 1) for c_ctx, c in
                                      zip(self.context_encoder.out_channels[-self.FUSED_LEVELS:], channels))
            for conv in self.fuse:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = target_slice(x.shape[-1], self.target)
        if not self.context_scale:
            return self.net(x)[:, :, t, t]
        x, context = x[:, :self.in_channels], x[:, self.in_channels:]
        features = self.net.encoder(x)
        context_features = self.context_encoder(context)[-self.FUSED_LEVELS:]
        # the window covers the central 1/scale of the context view, in both directions
        theta = torch.tensor([[1 / self.context_scale, 0, 0], [0, 1 / self.context_scale, 0]], device=x.device)
        for i, (conv, ctx) in enumerate(zip(self.fuse, context_features)):
            level = len(features) - self.FUSED_LEVELS + i
            grid = F.affine_grid(theta.expand(len(x), 2, 3), list(features[level].shape), align_corners=False)
            ctx = F.grid_sample(ctx.float(), grid.float(), align_corners=False).to(features[level].dtype)
            features[level] = features[level] + conv(ctx)
        return self.net.segmentation_head(self.net.decoder(features))[:, :, t, t]


class FlipRotateAverage(nn.Module):
    """Wraps a model to average its class probabilities over the 8 rotations/flips of the input (test-time
    augmentation). The output is a probability map, so argmax works as with logits."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        total = 0
        for flip in (False, True):
            xf = x.flip(3) if flip else x
            for k in range(4):  # predict on the transformed input, then undo the transform on the output
                y = self.model(xf.rot90(k, (2, 3))).float().softmax(1).rot90(-k, (2, 3))
                total = total + (y.flip(3) if flip else y)
        return total / 8


def load_model(path: str, device: str | torch.device = "cpu"):
    """A model from a checkpoint written by this script, in eval mode, plus the checkpoint's metadata."""
    checkpoint = torch.load(path, map_location=device)
    num_classes, target = len(checkpoint["classes"]), checkpoint["target"]
    in_channels = num_classes + 1 if checkpoint["input"] == "labels" else 3
    if checkpoint.get("model_kind") == "unet":
        model = UNet(in_channels, num_classes, target, checkpoint["encoder"], weights=None,
                     context_scale=checkpoint.get("context_scale", 0))
    else:
        model = PatchNet(in_channels, num_classes, target)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def class_weights(pixel_counts: list[int], power: float) -> torch.Tensor:
    """Loss weights proportional to frequency^-power (0 = even, 0.5 = inverse square root, 1 = inverse),
    scaled so the average weight per pixel stays 1 and the loss keeps its scale."""
    counts = torch.tensor(pixel_counts, dtype=torch.float64)
    weights = torch.where(counts > 0, (counts / counts.max()).clamp(min=1e-12) ** -power, torch.ones_like(counts))
    weights = weights * counts.sum() / (weights * counts).sum()
    return weights.float()


def print_table(columns: dict, class_names: list[str]):
    """Per-class IoU, one column per {name: summarize() result}."""
    print(f"\n{'class':<16}" + "".join(f"{name:>12}" for name in columns))
    for i, name in enumerate(class_names):
        print(f"{name:<16}" + "".join(f"{columns[k]['iou'][i]:>12.3f}" for k in columns))


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed precision on GPU: bfloat16 where supported (Ampere+, no loss scaling needed), else float16."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def summarize(c: torch.Tensor) -> dict:
    """Pixel accuracy, mIoU and per-class IoU from a confusion matrix (rows = truth, columns = prediction)."""
    tp = c.diag().float()
    iou = tp / (c.sum(0) + c.sum(1) - tp).float().clamp(min=1)
    present = c.sum(1) > 0
    iou[~present] = float("nan")  # class absent from these squares: no score rather than a misleading 0
    return {"acc": (tp.sum() / c.sum()).item(), "miou": iou[present].mean().item(), "iou": iou}


def merge_classes(c: torch.Tensor, merges: list[tuple[int, int]]) -> torch.Tensor:
    """Confusion matrix with each (source, destination) class pair counted as one class, the destination."""
    c = c.clone()
    for src, dst in merges:
        c[dst] += c[src]
        c[:, dst] += c[:, src]
        c[src], c[:, src] = 0, 0
    return c


def evaluate(model, val_batches, num_classes, target, input_kind, device, memory_format=torch.contiguous_format):
    """Confusion matrices on the target squares, for the model and, with label input, a baseline that
    copies, row by row, the nearest visible pixel to the left of the square."""
    model.eval()
    names = ["model", "copy_left"] if input_kind == "labels" else ["model"]
    confusion = {name: torch.zeros(num_classes, num_classes, dtype=torch.long, device=device) for name in names}
    dtype = amp_dtype(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        s = target_slice(target, min(target, SCORE))  # the scored square inside the target square
        for labels, images in val_batches:
            x, y = encode(labels, images, num_classes, target, input_kind, device, memory_format)
            t = target_slice(labels.shape[-1], target)
            y = y[:, s, s]
            preds = {"model": model(x).argmax(1)[:, s, s]}
            if input_kind == "labels":
                preds["copy_left"] = labels[:, t, t.start - 1:t.start].expand(-1, -1, target).to(device)
            scored = y != IGNORE
            for name, pred in preds.items():
                pred = pred.masked_fill(pred == IGNORE, 0)  # copy-left can copy an ignored pixel
                confusion[name] += torch.bincount(
                    (y[scored] * num_classes + pred[scored]), minlength=num_classes ** 2,
                ).view(num_classes, num_classes)
    model.train()
    return {name: c.cpu() for name, c in confusion.items()}


def evaluate_groups(model, groups, num_classes, target, input_kind, device, memory_format=torch.contiguous_format,
                    merges=()):
    """{"val"/"test": {area: batches}} -> pooled summaries per group and per-area mIoU, as
    ({"val": summary, "val copy_left": summary, "test": summary}, {"houten val": miou, ...}).
    `merges` lists (source, destination) class pairs scored as one class."""
    pooled, per_area = {}, {}
    for group, areas in groups.items():
        for area, batches in areas.items():
            confusion = evaluate(model, batches, num_classes, target, input_kind, device, memory_format)
            confusion = {name: merge_classes(c, merges) for name, c in confusion.items()}
            per_area[f"{area} {group}"] = summarize(confusion["model"])["miou"]
            for name, c in confusion.items():
                key = group if name == "model" else f"{group} {name}"
                pooled[key] = pooled.get(key, 0) + c
    return {key: summarize(c) for key, c in pooled.items()}, per_area


def tiles(lo: int, hi: int, target: int) -> list[tuple[int, int]]:
    """(start, skip) per target square covering [lo, hi): the last square is pulled back to end at hi, and
    `skip` is how much of it the previous square already covered."""
    starts = list(range(lo, hi - target + 1, target))
    if starts[-1] + target < hi:
        starts.append(hi - target)
    return [(b, a + target - b if i else 0) for i, (a, b) in enumerate(zip([lo] + starts, starts))]


def score_region(model, imagery_path: str, labels_path: str, rows: tuple[int, int], cols: tuple[int, int],
                 size: int, target: int, num_classes: int, device, batch: int = 32,
                 memory_format=torch.contiguous_format) -> torch.Tensor:
    """Confusion matrix over every pixel of labels[rows, cols], each predicted once, by the window around the
    target square that contains it. Needs (size - target) / 2 px of imagery around the region."""
    imagery, labels = np.load(imagery_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")
    margin, (c0, c1) = (size - target) // 2, cols
    confusion = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
    dtype = amp_dtype(device)
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for r, row_skip in tiles(*rows, target):
            strip = torch.from_numpy(np.array(imagery[r - margin:r + target + margin, c0 - margin:c1 + margin])).to(device)
            pred = torch.empty(target, c1 - c0, dtype=torch.long, device=device)
            col_tiles = tiles(c0, c1, target)
            for i in range(0, len(col_tiles), batch):
                chunk = col_tiles[i:i + batch]
                windows = torch.stack([strip[:, c - c0:c - c0 + size] for c, _ in chunk])
                out = model(encode_imagery(windows).contiguous(memory_format=memory_format)).argmax(1)
                for (c, skip), square in zip(chunk, out):
                    pred[:, c + skip - c0:c + target - c0] = square[:, skip:]
            truth = torch.from_numpy(np.array(labels[r + row_skip:r + target, c0:c1])).to(device).long()
            pred, scored = pred[row_skip:], truth != IGNORE
            confusion += torch.bincount(truth[scored] * num_classes + pred[scored], minlength=num_classes ** 2)
    model.train()
    return confusion.view(num_classes, num_classes).cpu()


def score_groups_full(model, areas: dict, size: int, target: int, num_classes: int, device,
                      memory_format=torch.contiguous_format, merges=()):
    """Like evaluate_groups, but over every pixel: the validation strip of each training area and the whole of
    each test area, less a MAX_SIZE / 2 border, the same region the evaluation windows are centred in.
    `areas` = {"val"/"test": {area: (labels_path, imagery_path)}}."""
    pooled, per_area = {}, {}
    for group, paths in areas.items():
        for area, (labels_path, imagery_path) in paths.items():
            height, width = np.load(labels_path, mmap_mode="r").shape
            top = int(height * (1 - VAL_FRACTION)) if group == "val" else 0
            rows, cols = (top + MAX_SIZE // 2, height - MAX_SIZE // 2), (MAX_SIZE // 2, width - MAX_SIZE // 2)
            c = score_region(model, imagery_path, labels_path, rows, cols, size, target, num_classes, device,
                             memory_format=memory_format)
            c = merge_classes(c, merges)
            per_area[f"{area} {group}"] = summarize(c)["miou"]
            pooled[group] = pooled.get(group, 0) + c
    return {key: summarize(c) for key, c in pooled.items()}, per_area


def report(summaries: dict, per_area: dict) -> str:
    groups = "  ".join(f"{key} acc {r['acc']:.3f} mIoU {r['miou']:.3f}" for key, r in summaries.items())
    areas = ", ".join(f"{key} {miou:.3f}" for key, miou in per_area.items()) if len(per_area) > 1 else ""
    return groups + (f"  | per area mIoU: {areas}" if areas else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=DEFAULTS, default="imagery")
    parser.add_argument("--areas", nargs="+", default=["houten"], help="training areas; their southern strip is val")
    parser.add_argument("--test-areas", nargs="*", default=[], help="areas held out entirely, scored as 'test'")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--size", type=int, help="input window side in px (default: 256 imagery, 160 labels)")
    parser.add_argument("--target", type=int, help="side of the central square that is scored "
                                                    "(and hidden, with label input); default 128 / 32")
    parser.add_argument("--no-augment", action="store_true", help="skip the random rotations/flips")
    parser.add_argument("--lighting", action="store_true",
                        help="also vary brightness, contrast, gamma and colour of training windows, with synthetic shadows")
    parser.add_argument("--context-scale", type=int, default=0,
                        help="U-Net only: add a context view this many times coarser (4 = 123 m around a 384 px "
                             "window); 0 = off")
    parser.add_argument("--model", choices=("dilated", "unet"), default="dilated")
    parser.add_argument("--encoder", default="resnet34", help="U-Net encoder (any segmentation_models_pytorch name)")
    parser.add_argument("--encoder-weights", default="imagenet", help="'imagenet', or 'none' to train from scratch")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=50, help="fixed validation batches per area")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="DataLoader processes sampling training windows; 0 samples in the main process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels-last", action="store_true",
                        help="NHWC memory layout, which tensor cores prefer; same results, often faster on GPU")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model; same results, faster after a slower first steps")
    parser.add_argument("--class-weight-power", type=float, default=0.0,
                        help="weight the loss by class frequency^-power: 0 = even, 0.25 = slight, 0.5 = strong")
    parser.add_argument("--evaluate", metavar="CHECKPOINT",
                        help="only score this checkpoint on the validation squares (uses its size/target), then exit")
    parser.add_argument("--tta", action="store_true",
                        help="with --evaluate: average predictions over the 8 rotations/flips (8x slower)")
    parser.add_argument("--merge", nargs="+", default=[], metavar="FROM=TO",
                        help="with --evaluate: score class FROM as TO, e.g. traffic_island=sidewalk")
    parser.add_argument("--ema", type=float, default=0.0, metavar="DECAY",
                        help="evaluate and save an exponential moving average of the weights, e.g. 0.9995 "
                             "(about the last 2,000 steps); 0 = off")
    parser.add_argument("--full-eval", action="store_true",
                        help="after training (or with --evaluate), also score every pixel of the validation strips "
                             "and test areas; imagery input without a context view only")
    parser.add_argument("--infer-target", type=int,
                        help="with --evaluate: predict this central square per window instead of the checkpoint's "
                             "target (smaller = more context per pixel, more windows)")
    parser.add_argument("--seed", type=int, default=0,
                        help="initialisation and training windows; validation windows stay fixed whatever the seed")
    parser.add_argument("--out", default="data/center_patch.pt")
    args = parser.parse_args()
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu")
        args.input, args.size, args.target = checkpoint["input"], checkpoint["size"], checkpoint["target"]
        args.target = args.infer_target or args.target
        args.context_scale = checkpoint.get("context_scale", 0)
    args.size = args.size or DEFAULTS[args.input][0]
    args.target = args.target or DEFAULTS[args.input][1]
    if not 1 <= args.target <= args.size - 2 or (args.size - args.target) % 2:
        parser.error("--target must be smaller than --size, with an even difference so it can be centred")

    if args.context_scale and (args.input != "imagery" or args.model != "unet" and not args.evaluate):
        parser.error("--context-scale needs imagery input and the U-Net")
    if args.full_eval and (args.input != "imagery" or args.context_scale):
        parser.error("--full-eval needs imagery input without a context view")
    overlap = set(args.areas) & set(args.test_areas)
    if overlap:
        parser.error(f"areas can't be both training and test: {', '.join(sorted(overlap))}")
    labels_path = lambda area: f"{args.data_dir}/{area}_classes_8cm"
    imagery_path = lambda area: f"{args.data_dir}/{area}_aerial_2026.npy" if args.input == "imagery" else None
    raster_meta = {}
    for area in args.areas + args.test_areas:
        with open(f"{labels_path(area)}.json") as file:
            raster_meta[area] = json.load(file)
    class_names = [name for _, name in sorted(raster_meta[args.areas[0]]["classes"].items(), key=lambda kv: int(kv[0]))]
    if any([n for _, n in sorted(m["classes"].items(), key=lambda kv: int(kv[0]))] != class_names
           for m in raster_meta.values()):
        parser.error("all areas must use the same class table; rerun rasterize.py")
    num_classes = len(class_names)
    if args.context_scale:
        for area in args.areas + args.test_areas:
            if not os.path.exists(context_path(imagery_path(area), args.context_scale)):
                print(f"building the {args.context_scale}x context imagery for {area}", flush=True)
                build_context(imagery_path(area), args.context_scale)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True  # fixed input sizes, so let cuDNN pick the fastest kernels
    groups = {}  # fixed batches, so evals are comparable between steps and runs
    for group, split, areas in (("val", "val", args.areas), ("test", "all", args.test_areas)):
        for area in areas:
            windows = Windows(f"{labels_path(area)}.npy", imagery_path(area), split, args.size, seed=1,
                              context_scale=args.context_scale)
            groups.setdefault(group, {})[area] = [windows.batch(args.batch) for _ in range(args.val_batches)]
    full_areas = {group: {a: (f"{labels_path(a)}.npy", imagery_path(a)) for a in areas}
                  for group, areas in (("val", args.areas), ("test", args.test_areas)) if areas}

    if args.evaluate:
        model, _ = load_model(args.evaluate, device)
        model.target = args.target
        if args.tta:
            model = FlipRotateAverage(model)
        merges = [tuple(class_names.index(name) for name in pair.split("=")) for pair in args.merge]
        summaries, per_area = evaluate_groups(model, groups, num_classes, args.target, args.input, device,
                                              merges=merges)
        print(f"{args.evaluate}: {report(summaries, per_area)}")
        print_table(summaries, class_names)
        if args.full_eval:
            summaries, per_area = score_groups_full(model, full_areas, args.size, args.target, num_classes, device,
                                                    merges=merges)
            print(f"{args.evaluate}, every pixel: {report(summaries, per_area)}")
            print_table(summaries, class_names)
        return
    train = DataLoader(
        TrainStream([(f"{labels_path(a)}.npy", imagery_path(a)) for a in args.areas], args.size,
                    augment=not args.no_augment, lighting=args.lighting, context_scale=args.context_scale,
                    seed=args.seed),
        batch_size=args.batch, num_workers=args.workers, pin_memory=cuda,
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,  # batches sit in /dev/shm
    )

    in_channels = num_classes + 1 if args.input == "labels" else 3
    if args.model == "unet":
        if args.size % 32:
            parser.error("--size must be a multiple of 32 for the U-Net")
        weights = None if args.encoder_weights == "none" else args.encoder_weights
        model = UNet(in_channels, num_classes, args.target, args.encoder, weights, args.context_scale).to(device)
    else:
        model = PatchNet(in_channels, num_classes, args.target).to(device)
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=memory_format)
    run_model = torch.compile(model) if args.compile else model  # `model` keeps clean state_dict keys for saving
    ema, eval_model, run_eval_model = None, model, run_model  # what is evaluated and saved
    if args.ema:  # BatchNorm statistics are averaged too, so the averaged model needs no recalibration
        ema = torch.optim.swa_utils.AveragedModel(model, use_buffers=True,
                                                  multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
        eval_model = ema.module
        run_eval_model = torch.compile(eval_model) if args.compile else eval_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps)
    pixel_counts = [sum(raster_meta[a]["pixel_counts"][name] for a in args.areas) for name in class_names]
    weights = class_weights(pixel_counts, args.class_weight_power)
    if args.class_weight_power:
        print("class weights: " + ", ".join(f"{n} {w:.2f}" for n, w in zip(class_names, weights.tolist())))
    weights = weights.to(device)
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)  # bfloat16 doesn't need loss scaling
    print(f"training {args.model} on {device} ({dtype or 'float32'}), {args.workers} sampling workers, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
          f"{', channels-last' if args.channels_last else ''}{', compiled' if args.compile else ''}", flush=True)

    meta = {"classes": class_names, "input": args.input, "model_kind": args.model,
            "encoder": args.encoder if args.model == "unet" else None, "size": args.size, "target": args.target,
            "class_weight_power": args.class_weight_power, "lighting": args.lighting,
            "context_scale": args.context_scale, "ema": args.ema, "seed": args.seed,
            "areas": args.areas, "test_areas": args.test_areas}
    start = time.time()
    for step, (labels, images) in enumerate(train, start=1):
        x, y = encode(labels, images, num_classes, args.target, args.input, device, memory_format)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = F.cross_entropy(run_model(x), y, weight=weights, ignore_index=IGNORE)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema:
            ema.update_parameters(model)

        if step % args.eval_every == 0 or step == args.steps:
            summaries, per_area = evaluate_groups(run_eval_model, groups, num_classes, args.target, args.input,
                                                  device, memory_format)
            print(f"step {step:5d}  loss {loss.item():.3f}  {report(summaries, per_area)}  "
                  f"{time.time() - start:.0f}s", flush=True)
            torch.save({**meta, "model": eval_model.state_dict(), "step": step, "val_miou": summaries["val"]["miou"],
                        "test_miou": summaries["test"]["miou"] if "test" in summaries else None}, args.out)
        if step == args.steps:
            break

    if ema:  # the raw weights at the end, for comparison
        raw, raw_per_area = evaluate_groups(run_model, groups, num_classes, args.target, args.input, device,
                                            memory_format)
        print(f"without EMA: {report(raw, raw_per_area)}", flush=True)
    print_table(summaries, class_names)
    if args.full_eval:
        start = time.time()
        summaries, per_area = score_groups_full(run_eval_model, full_areas, args.size, args.target, num_classes,
                                                device, memory_format)
        print(f"every pixel: {report(summaries, per_area)}  {time.time() - start:.0f}s", flush=True)
        print_table(summaries, class_names)
        if ema:
            raw, raw_per_area = score_groups_full(run_model, full_areas, args.size, args.target, num_classes,
                                                  device, memory_format)
            print(f"every pixel, without EMA: {report(raw, raw_per_area)}", flush=True)


if __name__ == "__main__":
    main()
