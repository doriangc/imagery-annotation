"""Predict a block of the map with a center_patch.py checkpoint and render photo | BGT | prediction | errors.

The block is tiled with the model's window: each window predicts only its central target square, so every pixel
gets the full context margin it was trained with. Windows step by the target size, or by less with --stride, in
which case overlapping squares are averaged; --tta also averages over rotations and flips.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from center_patch import IGNORE, VAL_FRACTION, build_context, context_window, encode_imagery, load_model
from rasterize import CLASSES

PALETTE = np.full((256, 3), 255, dtype=np.uint8)  # IGNORE shows as white
PALETTE[:len(CLASSES)] = [rgb for _, rgb in CLASSES.values()]


def predict_block(model, imagery, row: int, col: int, extent: int, size: int, target: int, stride: int | None = None,
                  tta: bool = False, batch: int = 8, context=None, context_scale: int = 0):
    """Class ids for imagery[row:row + extent, col:col + extent]; extent must be a multiple of stride.

    stride < target overlaps the windows' target squares and averages their class probabilities, so neighbouring
    windows can't disagree along a seam. tta also averages over the 8 rotations/flips of each window. Pixels need
    context_margin(size, target, stride) of imagery around the block. Models with a context view also need the
    downsampled imagery (`context`, from build_context) and its scale."""
    stride = stride or target
    margin, overlap = (size - target) // 2, target - stride
    span = extent + 2 * overlap  # squares start up to `overlap` before the block and end up to it after
    starts = [(r, c) for r in range(row - overlap, row + extent, stride) for c in range(col - overlap, col + extent, stride)]
    transforms = [(k, flip) for k in range(4) for flip in (False, True)] if tta else [(0, False)]
    probs = None
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            crops = [imagery[r - margin:r - margin + size, c - margin:c - margin + size] for r, c in chunk]
            if context is not None:
                crops = [np.concatenate([crop, context_window(context, context_scale, r - margin, c - margin, size)], 2)
                         for crop, (r, c) in zip(crops, chunk)]
            windows = encode_imagery(torch.from_numpy(np.stack(crops)))
            p = 0
            for k, flip in transforms:  # predict on the transformed window, then undo the transform on the output
                x = windows.flip(3) if flip else windows
                y = model(x.rot90(k, (2, 3))).softmax(1).rot90(-k, (2, 3))
                p = p + (y.flip(3) if flip else y)
            if probs is None:
                probs = torch.zeros(p.shape[1], span, span)
            for (r, c), square in zip(chunk, p):
                probs[:, r - row + overlap:r - row + overlap + target, c - col + overlap:c - col + overlap + target] += square
    return probs[:, overlap:overlap + extent, overlap:overlap + extent].argmax(0).numpy().astype(np.uint8)


def context_margin(size: int, target: int, stride: int | None = None) -> int:
    """Imagery needed around a block for predict_block."""
    return (size - target) // 2 + target - (stride or target)


def legend(classes: list[str], width: int) -> Image.Image:
    image = Image.new("RGB", (width, 28), "white")
    draw = ImageDraw.Draw(image)
    x = 6
    for i, name in enumerate(classes):
        draw.rectangle([x, 8, x + 12, 20], fill=tuple(int(v) for v in PALETTE[i]), outline="gray")
        draw.text((x + 16, 8), name, fill="black")
        x += 22 + int(draw.textlength(name))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="data/unet_imagenet.pt")
    parser.add_argument("--area", default="houten", help="area from areas.py")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--at", type=int, nargs=2, action="append", metavar=("ROW", "COL"),
                        help="top-left pixel of a block (repeatable); default: random road-rich blocks the model "
                             "never trained on (the val strip of a training area, anywhere in other areas)")
    parser.add_argument("--count", type=int, default=3, help="random blocks when --at isn't given")
    parser.add_argument("--extent", type=int, default=768, help="block side in px (multiple of the model's target)")
    parser.add_argument("--stride", type=int, help="step between windows (default: the model's target); "
                        "half the target overlaps windows and averages them")
    parser.add_argument("--tta", action="store_true", help="also average over the 8 rotations/flips")
    parser.add_argument("--out", default="data/vis")
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint)
    size, target, classes = checkpoint["size"], checkpoint["target"], checkpoint["classes"]
    stride = args.stride or target
    if args.extent % target or target % stride:
        parser.error(f"--extent must be a multiple of the model's target ({target}), which --stride must divide")
    labels = np.load(f"{args.data_dir}/{args.area}_classes_8cm.npy", mmap_mode="r")
    imagery_path = f"{args.data_dir}/{args.area}_aerial_2026.npy"
    imagery = np.load(imagery_path, mmap_mode="r")
    context_scale = checkpoint.get("context_scale", 0)
    context = np.load(build_context(imagery_path, context_scale), mmap_mode="r") if context_scale else None
    margin = context_margin(size, target, stride)
    height, width = labels.shape

    blocks = args.at
    if not blocks:  # only pixels the model never trained on
        rng = np.random.default_rng(0)
        pick = context_margin(size, target, 1)  # the same blocks whatever --stride is
        trained_on = args.area in checkpoint.get("areas", ["houten"])
        val_top = int(height * (1 - VAL_FRACTION)) if trained_on else 0
        blocks = []
        while len(blocks) < args.count:
            r = int(rng.integers(val_top + pick, height - pick - args.extent))
            c = int(rng.integers(pick, width - pick - args.extent))
            block = labels[r:r + args.extent:8, c:c + args.extent:8]
            if np.isin(block, [1, 2, 3, 4]).mean() > 0.25:  # road, bike lane, sidewalk, parking
                blocks.append((r, c))

    for r, c in blocks:
        if not (margin <= r <= height - margin - args.extent and margin <= c <= width - margin - args.extent):
            parser.error(f"block at {r}, {c} needs {margin} px of imagery around it inside the raster")
        photo = np.array(imagery[r:r + args.extent, c:c + args.extent])
        truth = np.array(labels[r:r + args.extent, c:c + args.extent])
        pred = predict_block(model, imagery, r, c, args.extent, size, target, stride, args.tta,
                             context=context, context_scale=context_scale)
        scored = truth != IGNORE
        wrong = (pred != truth) & scored
        errors = (photo * 0.45).astype(np.uint8)
        errors[wrong] = (255, 40, 40)
        errors[~scored] = (200, 200, 200)
        panels = np.concatenate([photo, PALETTE[truth], PALETTE[pred], errors], axis=1)
        gap = np.full((args.extent, 8, 3), 255, np.uint8)
        panels = np.concatenate(sum(([p, gap] for p in np.split(panels, 4, axis=1)), [])[:-1], axis=1)
        image = Image.new("RGB", (panels.shape[1], panels.shape[0] + 28), "white")
        image.paste(Image.fromarray(panels), (0, 0))
        image.paste(legend(classes, panels.shape[1]), (0, panels.shape[0]))
        path = f"{args.out}_{args.area}_r{r}_c{c}.png"
        image.save(path)
        print(f"{path}  accuracy {1 - wrong.sum() / max(scored.sum(), 1):.3f}  "
              + json.dumps({classes[k]: round(float((pred[truth == k] == k).mean()), 2)
                            for k in np.unique(truth) if k != IGNORE}))


if __name__ == "__main__":
    main()
