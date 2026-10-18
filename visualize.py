"""Predict a block of the map with a center_patch.py checkpoint and render photo | BGT | prediction | errors.

The block is tiled with the model's window: each window predicts only its central target square, and windows
step by the target size, so every pixel gets the full context margin it was trained with.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from center_patch import IGNORE, VAL_FRACTION, encode_imagery, load_model
from rasterize import CLASSES

PALETTE = np.full((256, 3), 255, dtype=np.uint8)  # IGNORE shows as white
PALETTE[:len(CLASSES)] = [rgb for _, rgb in CLASSES.values()]


def predict_block(model, imagery, row: int, col: int, extent: int, size: int, target: int, batch: int = 8):
    """Class ids for imagery[row:row + extent, col:col + extent]; extent must be a multiple of target."""
    margin = (size - target) // 2
    starts = [(r, c) for r in range(row, row + extent, target) for c in range(col, col + extent, target)]
    out = np.zeros((extent, extent), np.uint8)
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            windows = np.stack([imagery[r - margin:r - margin + size, c - margin:c - margin + size] for r, c in chunk])
            preds = model(encode_imagery(torch.from_numpy(windows))).argmax(1).numpy()
            for (r, c), pred in zip(chunk, preds):
                out[r - row:r - row + target, c - col:c - col + target] = pred
    return out


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
    parser.add_argument("--out", default="data/vis")
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint)
    size, target, classes = checkpoint["size"], checkpoint["target"], checkpoint["classes"]
    if args.extent % target:
        parser.error(f"--extent must be a multiple of the model's target ({target})")
    labels = np.load(f"{args.data_dir}/{args.area}_classes_8cm.npy", mmap_mode="r")
    imagery = np.load(f"{args.data_dir}/{args.area}_aerial_2026.npy", mmap_mode="r")
    margin = (size - target) // 2
    height, width = labels.shape

    blocks = args.at
    if not blocks:  # only pixels the model never trained on
        rng = np.random.default_rng(0)
        trained_on = args.area in checkpoint.get("areas", ["houten"])
        val_top = int(height * (1 - VAL_FRACTION)) if trained_on else 0
        blocks = []
        while len(blocks) < args.count:
            r = int(rng.integers(val_top + margin, height - margin - args.extent))
            c = int(rng.integers(margin, width - margin - args.extent))
            block = labels[r:r + args.extent:8, c:c + args.extent:8]
            if np.isin(block, [1, 2, 3, 4]).mean() > 0.25:  # road, bike lane, sidewalk, parking
                blocks.append((r, c))

    for r, c in blocks:
        if not (margin <= r <= height - margin - args.extent and margin <= c <= width - margin - args.extent):
            parser.error(f"block at {r}, {c} needs {margin} px of imagery around it inside the raster")
        photo = np.array(imagery[r:r + args.extent, c:c + args.extent])
        truth = np.array(labels[r:r + args.extent, c:c + args.extent])
        pred = predict_block(model, imagery, r, c, args.extent, size, target)
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
