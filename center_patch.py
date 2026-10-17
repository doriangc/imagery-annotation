"""Predict the BGT classes of a central target square from a larger square window around it.

Step one towards predicting from imagery: here the input is the class raster itself with the target
square hidden, so the model has to infer it from context. Swapping the input for image channels later
leaves the sampling, the output and the evaluation unchanged (and then nothing needs hiding).

The model predicts every pixel, but only the target square is scored: its pixels all have at least
`(size - target) / 2` px of context on every side, which should cover the model's receptive field.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VAL_FRACTION = 0.15  # the southern strip of the raster is held out


class Windows:
    """Random square windows from the class raster, split by north/south strip so train and val never overlap."""

    def __init__(self, path: str, split: str, size: int, seed: int = 0):
        self.raster = np.load(path, mmap_mode="r")
        self.size = size
        height, width = self.raster.shape
        val_top = int(height * (1 - VAL_FRACTION))
        if split == "train":
            self.row_range = (0, val_top - size)
        else:
            self.row_range = (val_top, height - size)
        self.col_range = (0, width - size)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        row = self.rng.integers(*self.row_range)
        col = self.rng.integers(*self.col_range)
        return np.array(self.raster[row:row + self.size, col:col + self.size])

    def batch(self, n: int) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.sample() for _ in range(n)])).long()


def target_slice(size: int, target: int) -> slice:
    """The central `target` px of a `size` px window, along one axis."""
    start = (size - target) // 2
    return slice(start, start + target)


def encode(windows: torch.Tensor, num_classes: int, target: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot input with the target square blanked plus a mask channel; the target is that square's labels."""
    t = target_slice(windows.shape[-1], target)
    x = F.one_hot(windows, num_classes).permute(0, 3, 1, 2).float()
    x[:, :, t, t] = 0
    mask = torch.zeros_like(x[:, :1])
    mask[:, :, t, t] = 1
    return torch.cat([x, mask], dim=1), windows[:, t, t]


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


def evaluate(model, val_batches, num_classes, target):
    """Pixel accuracy and per-class IoU on the target square, for the model and a baseline that copies,
    row by row, the nearest visible pixel to the left of the square."""
    model.eval()
    confusion = {"model": torch.zeros(num_classes, num_classes, dtype=torch.long),
                 "copy_left": torch.zeros(num_classes, num_classes, dtype=torch.long)}
    with torch.no_grad():
        for windows in val_batches:
            x, y = encode(windows, num_classes, target)
            t = target_slice(windows.shape[-1], target)
            preds = {"model": model(x).argmax(1),
                     "copy_left": windows[:, t, t.start - 1:t.start].expand(-1, -1, target)}
            for name, pred in preds.items():
                confusion[name] += torch.bincount(
                    (y * num_classes + pred).ravel(), minlength=num_classes ** 2,
                ).view(num_classes, num_classes)
    model.train()
    results = {}
    for name, c in confusion.items():
        tp = c.diag().float()
        iou = tp / (c.sum(0) + c.sum(1) - tp).float().clamp(min=1)
        present = c.sum(1) > 0
        iou[~present] = float("nan")  # class absent from the val strip: no score rather than a misleading 0
        results[name] = {"acc": (tp.sum() / c.sum()).item(), "miou": iou[present].mean().item(), "iou": iou}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default="data/houten_classes_8cm")
    parser.add_argument("--size", type=int, default=160, help="input window side in px")
    parser.add_argument("--target", type=int, default=32, help="side of the central square that is hidden and scored")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--out", default="data/center_patch.pt")
    args = parser.parse_args()
    if not 1 <= args.target <= args.size - 2 or (args.size - args.target) % 2:
        parser.error("--target must be smaller than --size, with an even difference so it can be centred")

    with open(f"{args.raster}.json") as file:
        class_names = [name for _, name in sorted(json.load(file)["classes"].items(), key=lambda kv: int(kv[0]))]
    num_classes = len(class_names)

    torch.manual_seed(0)
    train = Windows(f"{args.raster}.npy", "train", args.size, seed=0)
    val = Windows(f"{args.raster}.npy", "val", args.size, seed=1)
    val_batches = [val.batch(args.batch) for _ in range(args.val_batches)]  # fixed, so evals are comparable

    model = PatchNet(num_classes + 1, num_classes, args.target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps)

    start = time.time()
    for step in range(1, args.steps + 1):
        x, y = encode(train.batch(args.batch), num_classes, args.target)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(model, val_batches, num_classes, args.target)
            print(f"step {step:5d}  loss {loss.item():.3f}  "
                  f"val acc {r['model']['acc']:.3f} mIoU {r['model']['miou']:.3f}  "
                  f"(copy-left acc {r['copy_left']['acc']:.3f} mIoU {r['copy_left']['miou']:.3f})  "
                  f"{time.time() - start:.0f}s", flush=True)

    print(f"\n{'class':<16}{'model IoU':>10}{'copy-left':>10}")
    for i, name in enumerate(class_names):
        print(f"{name:<16}{r['model']['iou'][i]:>10.3f}{r['copy_left']['iou'][i]:>10.3f}")
    torch.save({"model": model.state_dict(), "classes": class_names,
                "size": args.size, "target": args.target}, args.out)


if __name__ == "__main__":
    main()
