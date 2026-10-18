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

from areas import AREAS
from label_report import bike_streets

CRS = "EPSG:28992"
TILE = 300  # px per side when burning (30 m at 10 cm); bounds peak memory
PREVIEW_STEP = 20  # 2 m/px preview; must divide TILE
IGNORE = 255  # pixels left out of training and scoring, e.g. fietsstraten, which municipalities label inconsistently
BIKE_STREET_MIN_OVERLAP_M = 3.0  # road polygons a fietsstraat runs along for at least this long are ignored

# id: (name, preview RGB)
CLASSES = {
    0: ("other", (0, 0, 0)),  # BGT is gapless at ground level, so this is buildings, water, yards, bare ground, ...
    1: ("road", (90, 90, 90)),
    2: ("bike_lane", (200, 60, 50)),
    3: ("sidewalk", (230, 210, 170)),
    4: ("parking", (70, 110, 200)),
    5: ("landscaping", (60, 170, 70)),
    6: ("unused_6", (160, 120, 200)),  # was driveway, now merged into sidewalk (not every town maps inrit); id kept
                                     # so class ids and earlier checkpoints stay compatible
    7: ("traffic_island", (240, 160, 40)),
    8: ("speed_bump", (255, 240, 0)),
    9: ("rail", (120, 60, 30)),
    10: ("unused_10", (150, 150, 120)),  # was verge_paved; verges are now split into sidewalk/landscaping, since
                                        # towns map them very differently (Amsterdam: mostly unpaved tree pits)
    11: ("agri_natural", (150, 200, 110)),
    12: ("stairs", (255, 120, 180)),
}
CLASS_ID = {name: i for i, (name, _) in CLASSES.items()}

PAVED = {"open verharding", "gesloten verharding"}
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
    "inrit": "sidewalk",  # driveways; some municipalities don't map them separately
    "overweg": "road",  # level crossing
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
        # berm (verge): paved looks like sidewalk from above; planted, unpaved or half-paved like landscaping
        return "sidewalk" if props["fysiek_voorkomen"] in PAVED else "landscaping"
    if collection == "begroeidterreindeel":
        return "landscaping" if props["fysiek_voorkomen"] == "groenvoorziening" else "agri_natural"
    return None


def load(path: str, bike_street_lines: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Geometries in RD and their class ids, sorted so later entries should be burned on top.
    Road polygons under the given fietsstraat centrelines (lon/lat point lists) get IGNORE."""
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
            rows.append((key, f["geometry"], CLASS_ID[name], collection))
    rows.sort(key=lambda r: r[0])
    project = lambda g: shapely.transform(g, lambda xy: np.column_stack(to_rd.transform(xy[:, 0], xy[:, 1])))
    geoms = project(shapely.from_geojson([json.dumps(r[1]) for r in rows]))
    class_ids = np.array([r[2] for r in rows], dtype=np.uint8)

    if bike_street_lines:
        roads = np.flatnonzero([r[3] == "wegdeel" for r in rows])
        lines = project(np.array([shapely.LineString(l) for l in bike_street_lines if len(l) > 1], dtype=object))
        line_idx, road_idx = shapely.STRtree(geoms[roads]).query(lines, predicate="intersects")
        overlap = np.zeros(len(roads))
        np.add.at(overlap, road_idx, shapely.length(shapely.intersection(lines[line_idx], geoms[roads][road_idx])))
        class_ids[roads[overlap >= BIKE_STREET_MIN_OVERLAP_M]] = IGNORE
        print(f"{len(lines)} OSM fietsstraten -> {(overlap >= BIKE_STREET_MIN_OVERLAP_M).sum()} road polygons ignored")
    return geoms, class_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", nargs="?", default="houten", choices=sorted(AREAS),
                        help="area from areas.py; reads data/<area>_bgt.pkl and crops to its bbox, since fetched "
                             "polygons stick out past it")
    parser.add_argument("--res", type=float, default=0.08, help="metres per pixel; 0.08 matches PDOK's 8 cm aerial photos")
    parser.add_argument("--out", help="output prefix (default: data/<area>_classes_8cm)")
    parser.add_argument("--keep-bike-streets", action="store_true",
                        help="label fietsstraten with their BGT class instead of IGNORE")
    args = parser.parse_args()
    args.out = args.out or f"data/{args.area}_classes_8cm"

    lines = None if args.keep_bike_streets else bike_streets(args.area, "data")
    geoms, class_ids = load(f"data/{args.area}_bgt.pkl", lines)
    to_rd = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    west, south, east, north = to_rd.transform_bounds(*AREAS[args.area])
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
    counts = np.zeros(256, dtype=np.int64)
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
            counts += np.bincount(tile.ravel(), minlength=256)
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
        "ignore": IGNORE,
        "bike_streets_ignored": not args.keep_bike_streets,
        "pixel_counts": {**{name: int(counts[i]) for i, (name, _) in CLASSES.items()}, "ignore": int(counts[IGNORE])},
    }
    with open(f"{args.out}.json", "w") as file:
        json.dump(meta, file, indent=2)

    palette = np.full((256, 3), 255, dtype=np.uint8)  # IGNORE previews as white
    palette[:len(CLASSES)] = [rgb for _, rgb in CLASSES.values()]
    Image.fromarray(palette[preview]).save(f"{args.out}_preview.png")
    print("\n" + json.dumps(meta["pixel_counts"], indent=2))


if __name__ == "__main__":
    main()
