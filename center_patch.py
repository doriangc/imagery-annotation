"""Predict the BGT classes of a central target square from a larger square window around it.

Two inputs:
- imagery (default): the aerial photo crop, on the same pixel grid as the labels.
- labels: the class raster itself with the target square hidden, so the model has to infer it from context.
  This was the first step for setting up the code, and stays as a regression test with a copy-left baseline.

The model predicts every pixel, but only the target square is scored: its pixels all have at least
`(size - target) / 2` px of context on every side, which should cover the model's receptive field.
Training windows get a random 90° rotation and flip; validation windows stay north-up.
Runs on a GPU when one is available, with windows sampled by DataLoader worker processes.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

VAL_FRACTION = 0.15  # the southern strip of the raster is held out
DEFAULTS = {"imagery": (256, 128), "labels": (160, 32)}  # (size, target) per input
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # so pretrained encoders can drop in later
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Windows:
    """Random square windows (labels, and optionally the matching imagery), split by north/south strip
    so train and val never overlap."""

    def __init__(self, labels_path: str, imagery_path: str | None, split: str, size: int, seed: int = 0):
        self.labels = np.load(labels_path, mmap_mode="r")
        self.imagery = np.load(imagery_path, mmap_mode="r") if imagery_path else None
        if self.imagery is not None:
            assert self.imagery.shape[:2] == self.labels.shape, "imagery and labels must share the pixel grid"
        self.size = size
        height, width = self.labels.shape
        val_top = int(height * (1 - VAL_FRACTION))
        if split == "train":
            self.row_range = (0, val_top - size)
        else:
            self.row_range = (val_top, height - size)
        self.col_range = (0, width - size)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """(size, size) labels and (size, size, 3) uint8 imagery; the imagery is empty without an imagery path."""
        row = self.rng.integers(*self.row_range)
        col = self.rng.integers(*self.col_range)
        labels = np.array(self.labels[row:row + self.size, col:col + self.size])
        if self.imagery is None:
            return labels, np.zeros((0, 0, 3), np.uint8)
        return labels, np.array(self.imagery[row:row + self.size, col:col + self.size])

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


class TrainStream(IterableDataset):
    """Endless augmented training windows. Each DataLoader worker samples with its own seed."""

    def __init__(self, labels_path: str, imagery_path: str | None, size: int, augment: bool, seed: int = 0):
        self.args = (labels_path, imagery_path, "train", size)
        self.augment = augment
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        seed = self.seed + (info.id if info else 0)
        windows = Windows(*self.args, seed=seed)
        rng = np.random.default_rng(seed + 1000)
        while True:
            labels, image = windows.sample()
            if self.augment:
                labels, image = augment(labels, image, rng)
            yield labels, image


def target_slice(size: int, target: int) -> slice:
    """The central `target` px of a `size` px window, along one axis."""
    start = (size - target) // 2
    return slice(start, start + target)


def encode_labels(labels: torch.Tensor, num_classes: int, target: int) -> torch.Tensor:
    """One-hot input with the target square blanked plus a mask channel."""
    t = target_slice(labels.shape[-1], target)
    x = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
    x[:, :, t, t] = 0
    mask = torch.zeros_like(x[:, :1])
    mask[:, :, t, t] = 1
    return torch.cat([x, mask], dim=1)


def encode_imagery(images: torch.Tensor) -> torch.Tensor:
    """(n, size, size, 3) uint8 -> ImageNet-normalised (n, 3, size, size) float."""
    mean, std = IMAGENET_MEAN.to(images.device), IMAGENET_STD.to(images.device)
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
    """U-Net with a timm/torchvision encoder (ImageNet-pretrained or not), cropped to the target square."""

    def __init__(self, in_channels: int, num_classes: int, target: int, encoder: str, weights: str | None):
        super().__init__()
        import segmentation_models_pytorch as smp  # only needed for this model
        self.target = target
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=in_channels, classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = target_slice(x.shape[-1], self.target)
        return self.net(x)[:, :, t, t]


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed precision on GPU: bfloat16 where supported (Ampere+, no loss scaling needed), else float16."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def evaluate(model, val_batches, num_classes, target, input_kind, device, memory_format=torch.contiguous_format):
    """Pixel accuracy and per-class IoU on the target square. With label input, also for a baseline that
    copies, row by row, the nearest visible pixel to the left of the square."""
    model.eval()
    names = ["model", "copy_left"] if input_kind == "labels" else ["model"]
    confusion = {name: torch.zeros(num_classes, num_classes, dtype=torch.long, device=device) for name in names}
    dtype = amp_dtype(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for labels, images in val_batches:
            x, y = encode(labels, images, num_classes, target, input_kind, device, memory_format)
            t = target_slice(labels.shape[-1], target)
            preds = {"model": model(x).argmax(1)}
            if input_kind == "labels":
                preds["copy_left"] = labels[:, t, t.start - 1:t.start].expand(-1, -1, target).to(device)
            for name, pred in preds.items():
                confusion[name] += torch.bincount(
                    (y * num_classes + pred).ravel(), minlength=num_classes ** 2,
                ).view(num_classes, num_classes)
    model.train()
    results = {}
    for name, c in confusion.items():
        c = c.cpu()
        tp = c.diag().float()
        iou = tp / (c.sum(0) + c.sum(1) - tp).float().clamp(min=1)
        present = c.sum(1) > 0
        iou[~present] = float("nan")  # class absent from the val strip: no score rather than a misleading 0
        results[name] = {"acc": (tp.sum() / c.sum()).item(), "miou": iou[present].mean().item(), "iou": iou}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=DEFAULTS, default="imagery")
    parser.add_argument("--raster", default="data/houten_classes_8cm")
    parser.add_argument("--imagery", default="data/houten_aerial_2026.npy")
    parser.add_argument("--size", type=int, help="input window side in px (default: 256 imagery, 160 labels)")
    parser.add_argument("--target", type=int, help="side of the central square that is scored "
                                                    "(and hidden, with label input); default 128 / 32")
    parser.add_argument("--no-augment", action="store_true", help="skip the random rotations/flips")
    parser.add_argument("--model", choices=("dilated", "unet"), default="dilated")
    parser.add_argument("--encoder", default="resnet34", help="U-Net encoder (any segmentation_models_pytorch name)")
    parser.add_argument("--encoder-weights", default="imagenet", help="'imagenet', or 'none' to train from scratch")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="DataLoader processes sampling training windows; 0 samples in the main process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels-last", action="store_true",
                        help="NHWC memory layout, which tensor cores prefer; same results, often faster on GPU")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model; same results, faster after a slower first steps")
    parser.add_argument("--out", default="data/center_patch.pt")
    args = parser.parse_args()
    args.size = args.size or DEFAULTS[args.input][0]
    args.target = args.target or DEFAULTS[args.input][1]
    if not 1 <= args.target <= args.size - 2 or (args.size - args.target) % 2:
        parser.error("--target must be smaller than --size, with an even difference so it can be centred")

    with open(f"{args.raster}.json") as file:
        class_names = [name for _, name in sorted(json.load(file)["classes"].items(), key=lambda kv: int(kv[0]))]
    num_classes = len(class_names)

    torch.manual_seed(0)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True  # fixed input sizes, so let cuDNN pick the fastest kernels
    imagery = args.imagery if args.input == "imagery" else None
    val = Windows(f"{args.raster}.npy", imagery, "val", args.size, seed=1)
    val_batches = [val.batch(args.batch) for _ in range(args.val_batches)]  # fixed, so evals are comparable
    train = DataLoader(
        TrainStream(f"{args.raster}.npy", imagery, args.size, augment=not args.no_augment),
        batch_size=args.batch, num_workers=args.workers, pin_memory=cuda,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
    )

    in_channels = num_classes + 1 if args.input == "labels" else 3
    if args.model == "unet":
        if args.size % 32:
            parser.error("--size must be a multiple of 32 for the U-Net")
        weights = None if args.encoder_weights == "none" else args.encoder_weights
        model = UNet(in_channels, num_classes, args.target, args.encoder, weights).to(device)
    else:
        model = PatchNet(in_channels, num_classes, args.target).to(device)
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=memory_format)
    run_model = torch.compile(model) if args.compile else model  # `model` keeps clean state_dict keys for saving
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps)
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)  # bfloat16 doesn't need loss scaling
    print(f"training {args.model} on {device} ({dtype or 'float32'}), {args.workers} sampling workers, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
          f"{', channels-last' if args.channels_last else ''}{', compiled' if args.compile else ''}", flush=True)

    meta = {"classes": class_names, "input": args.input, "model_kind": args.model,
            "encoder": args.encoder if args.model == "unet" else None, "size": args.size, "target": args.target}
    start = time.time()
    for step, (labels, images) in enumerate(train, start=1):
        x, y = encode(labels, images, num_classes, args.target, args.input, device, memory_format)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = F.cross_entropy(run_model(x), y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(run_model, val_batches, num_classes, args.target, args.input, device, memory_format)
            baseline = (f"  (copy-left acc {r['copy_left']['acc']:.3f} mIoU {r['copy_left']['miou']:.3f})"
                        if "copy_left" in r else "")
            print(f"step {step:5d}  loss {loss.item():.3f}  "
                  f"val acc {r['model']['acc']:.3f} mIoU {r['model']['miou']:.3f}{baseline}  "
                  f"{time.time() - start:.0f}s", flush=True)
            torch.save({**meta, "model": model.state_dict(), "step": step, "val_miou": r["model"]["miou"]}, args.out)
        if step == args.steps:
            break

    print(f"\n{'class':<16}" + "".join(f"{name:>10}" for name in r))
    for i, name in enumerate(class_names):
        print(f"{name:<16}" + "".join(f"{r[k]['iou'][i]:>10.3f}" for k in r))


if __name__ == "__main__":
    main()
