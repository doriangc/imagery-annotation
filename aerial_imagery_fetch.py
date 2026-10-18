"""Download PDOK aerial photos on exactly the pixel grid of a label raster written by rasterize.py.

Writes <out>.npy (memory-mappable uint8, rows x cols x RGB, row 0 = north) and <out>.json. Pixel (r, c)
covers the same ground as pixel (r, c) of the label raster. Interrupted downloads resume where they stopped.
"""
import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from PIL import Image

WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
NATIVE_RES = 0.08  # the HR layers' grid; the WMS snaps requests to it instead of resampling
TILE = 2500  # the WMS MaxWidth/MaxHeight
WORKERS = 4
RETRIES = 5


def fetch_tile(client: httpx.Client, layer: str, bbox: tuple, width: int, height: int) -> np.ndarray:
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer, "styles": "",
        "crs": "EPSG:28992", "bbox": ",".join(f"{v:.3f}" for v in bbox),
        "width": width, "height": height, "format": "image/jpeg",
    }
    for attempt in range(RETRIES):
        try:
            response = client.get(WMS, params=params)
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("image/"):
                raise ValueError(f"WMS returned {response.headers.get('content-type')}: {response.text[:200]}")
            image = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
            if image.shape != (height, width, 3):
                raise ValueError(f"expected {height}x{width}, got {image.shape}")
            return image
        except (httpx.HTTPError, ValueError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", nargs="?", default="houten", help="area from areas.py")
    parser.add_argument("--grid", help="label raster metadata from rasterize.py (default: data/<area>_classes_8cm.json)")
    parser.add_argument("--layer", default="2026_orthoHR", help="WMS layer; a fixed year keeps the dataset reproducible")
    parser.add_argument("--out", help="output prefix (default: data/<area>_aerial_2026)")
    args = parser.parse_args()
    args.grid = args.grid or f"data/{args.area}_classes_8cm.json"
    args.out = args.out or f"data/{args.area}_aerial_2026"

    with open(args.grid) as file:
        grid = json.load(file)
    res = grid["resolution_m"]
    west, north = grid["origin_top_left"]
    height, width = grid["shape"]
    off_grid = [v for v in (west, north) if abs(v / NATIVE_RES - round(v / NATIVE_RES)) > 1e-6]
    if res != NATIVE_RES or off_grid:
        parser.error(f"label grid ({res} m, origin {west}, {north}) is not on the photos' {NATIVE_RES} m grid; "
                     "rerun rasterize.py, which snaps to it")

    tiles = [(row, col) for row in range(0, height, TILE) for col in range(0, width, TILE)]
    progress_path = f"{args.out}.progress.json"
    try:
        with open(progress_path) as file:
            done = {tuple(t) for t in json.load(file)}
        out = np.load(f"{args.out}.npy", mmap_mode="r+")
        assert out.shape == (height, width, 3)
    except (FileNotFoundError, AssertionError):
        done = set()
        out = np.lib.format.open_memmap(f"{args.out}.npy", mode="w+", dtype=np.uint8, shape=(height, width, 3))
    todo = [t for t in tiles if t not in done]
    print(f"{width} x {height} px ({out.nbytes / 1e9:.1f} GB), {len(tiles)} tiles, {len(todo)} to fetch")

    def work(tile):
        row, col = tile
        h, w = min(TILE, height - row), min(TILE, width - col)
        x0, y1 = west + col * res, north - row * res
        return tile, fetch_tile(client, args.layer, (x0, y1 - h * res, x0 + w * res, y1), w, h)

    client = httpx.Client(timeout=120, headers={"User-Agent": "road-centerline-annotator/0.1"})
    start = time.time()
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            for i, ((row, col), image) in enumerate(pool.map(work, todo), start=1):
                out[row:row + image.shape[0], col:col + image.shape[1]] = image
                out.flush()
                done.add((row, col))
                with open(progress_path, "w") as file:
                    json.dump(sorted(done), file)
                print(f"  {i}/{len(todo)} tiles, {time.time() - start:.0f}s", end="\r", flush=True)
    finally:
        client.close()

    with open(f"{args.out}.json", "w") as file:
        json.dump({"layer": args.layer, "wms": WMS, "grid": args.grid, "crs": grid["crs"], "resolution_m": res,
                   "origin_top_left": grid["origin_top_left"], "shape": [height, width, 3],
                   "fetched": time.strftime("%Y-%m-%d")}, file, indent=2)
    print(f"\ndone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
