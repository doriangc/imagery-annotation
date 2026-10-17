"""Burn pickled BGT features (from bgt_fetch.py) into a uint8 class-id raster in RD New (EPSG:28992).

Writes <out>.npy (memory-mappable, row 0 = north) and <out>.json (geotransform + class table),
plus a downsampled <out>_preview.png for eyeballing.
"""
import argparse
import json
import pickle

import numpy as np
import pyproj
import shapely
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_origin

CRS = "EPSG:28992"
TILE = 300  # px per side when burning (30 m at 10 cm); bounds peak memory
PREVIEW_STEP = 20  # 2 m/px preview; must divide TILE
HOUTEN_BBOX = (5.1400516, 52.008016, 5.20152, 52.0446192)  # same as bgt_fetch.py

# id: (name, preview RGB)
CLASSES = {
    0: ("other", (0, 0, 0)),  # BGT is gapless at ground level, so this is buildings, water, yards, bare ground, ...
    1: ("road", (90, 90, 90)),
    2: ("bike_lane", (200, 60, 50)),
    3: ("sidewalk", (230, 210, 170)),
    4: ("parking", (70, 110, 200)),
    5: ("landscaping", (60, 170, 70)),
    6: ("driveway", (160, 120, 200)),
    7: ("traffic_island", (240, 160, 40)),
    8: ("speed_bump", (255, 240, 0)),
    9: ("rail", (120, 60, 30)),
    10: ("verge_paved", (150, 150, 120)),
    11: ("agri_natural", (150, 200, 110)),
    12: ("stairs", (255, 120, 180)),
}
CLASS_ID = {name: i for i, (name, _) in CLASSES.items()}

WEGDEEL_FUNCTIE = {
    "rijbaan lokale weg": "road",
    "rijbaan regionale weg": "road",
    "rijbaan autosnelweg": "road",
    "woonerf": "road",
    "OV-baan": "road",
    "fietspad": "bike_lane",
    "voetpad": "sidewalk",
    "voetgangersgebied": "sidewalk",
    "voetpad op trap": "stairs",
    "parkeervlak": "parking",
    "inrit": "driveway",
    "spoorbaan": "rail",
}


def classify(collection: str, props: dict) -> str | None:
    if collection == "wegdeel":
        if props["plus_functie"] == "verkeersdrempel":
            return "speed_bump"
        return WEGDEEL_FUNCTIE.get(props["functie"])
    if collection == "ondersteunendwegdeel":
        if props["functie"] == "verkeerseiland":
            return "traffic_island"
        return "landscaping" if props["fysiek_voorkomen"] == "groenvoorziening" else "verge_paved"
    if collection == "begroeidterreindeel":
        return "landscaping" if props["fysiek_voorkomen"] == "groenvoorziening" else "agri_natural"
    return None


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Geometries in RD and their class ids, sorted so later entries should be burned on top."""
    with open(path, "rb") as file:
        data = pickle.load(file)
    to_rd = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    # Within one height level BGT is a partition; across levels the higher one is what you see from above.
    collection_order = {"begroeidterreindeel": 0, "ondersteunendwegdeel": 1, "wegdeel": 2}
    rows = []
    for collection, features in data.items():
        for f in features:
            name = classify(collection, f["properties"])
            if name is None:
                continue
            key = (f["properties"]["relatieve_hoogteligging"] or 0, collection_order.get(collection, 0))
            rows.append((key, f["geometry"], CLASS_ID[name]))
    rows.sort(key=lambda r: r[0])
    geoms = shapely.from_geojson([json.dumps(g) for _, g, _ in rows])
    geoms = shapely.transform(geoms, lambda xy: np.column_stack(to_rd.transform(xy[:, 0], xy[:, 1])))
    return geoms, np.array([c for _, _, c in rows], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle", nargs="?", default="data/houten_bgt.pkl")
    parser.add_argument("--out", default="data/houten_classes_8cm")
    parser.add_argument("--res", type=float, default=0.08, help="metres per pixel; 0.08 matches PDOK's 8 cm aerial photos")
    parser.add_argument("--bbox", type=float, nargs=4, default=HOUTEN_BBOX, metavar=("W", "S", "E", "N"),
                        help="lon/lat extent to render; fetched polygons stick out past the fetch bbox")
    args = parser.parse_args()

    geoms, class_ids = load(args.pickle)
    to_rd = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    west, south, east, north = to_rd.transform_bounds(*args.bbox)
    # Even metres sit on the 8 cm grid of PDOK's aerial photos (multiples of 0.08 m), which the WMS snaps to
    west, south = np.floor(np.array([west, south]) / 2) * 2
    east, north = np.ceil([east, north])
    east = west + np.ceil((east - west) / args.res) * args.res  # whole pixels, so the extent is exact for WMS requests
    north = south + np.ceil((north - south) / args.res) * args.res
    width, height = int(round((east - west) / args.res)), int(round((north - south) / args.res))
    transform = from_origin(west, north, args.res, args.res)
    print(f"{len(geoms)} polygons -> {width} x {height} px ({width * height / 1e9:.2f} GB)")

    out = np.lib.format.open_memmap(f"{args.out}.npy", mode="w+", dtype=np.uint8, shape=(height, width))
    preview = np.zeros((-(-height // PREVIEW_STEP), -(-width // PREVIEW_STEP)), dtype=np.uint8)
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    tree = shapely.STRtree(geoms)
    for row in range(0, height, TILE):
        for col in range(0, width, TILE):
            h, w = min(TILE, height - row), min(TILE, width - col)
            tile_transform = transform * transform.translation(col, row)
            x0, y1 = tile_transform * (0, 0)
            x1, y0 = tile_transform * (w, h)
            hits = np.sort(tree.query(shapely.box(x0, y0, x1, y1)))  # keep burn order
            if not len(hits):
                counts[0] += h * w
                continue
            tile = rasterize(
                zip(geoms[hits], class_ids[hits]), out_shape=(h, w), transform=tile_transform, dtype=np.uint8,
            )
            out[row:row + h, col:col + w] = tile
            counts += np.bincount(tile.ravel(), minlength=len(CLASSES))
            small = tile[::PREVIEW_STEP, ::PREVIEW_STEP]
            r, c = row // PREVIEW_STEP, col // PREVIEW_STEP
            preview[r:r + small.shape[0], c:c + small.shape[1]] = small
        out.flush()  # write back dirty pages per tile row so they don't pile up
        print(f"  rows {row}/{height}", end="\r")
    del out
    meta = {
        "crs": CRS,
        "resolution_m": args.res,
        "origin_top_left": [float(west), float(north)],
        "shape": [height, width],
        "transform": list(transform)[:6],
        "classes": {i: name for i, (name, _) in CLASSES.items()},
        "pixel_counts": {CLASSES[i][0]: int(n) for i, n in enumerate(counts)},
    }
    with open(f"{args.out}.json", "w") as file:
        json.dump(meta, file, indent=2)

    palette = np.array([rgb for _, rgb in CLASSES.values()], dtype=np.uint8)
    Image.fromarray(palette[preview]).save(f"{args.out}_preview.png")
    print("\n" + json.dumps(meta["pixel_counts"], indent=2))


if __name__ == "__main__":
    main()
