"""Fetch US areas for the road-only model: photos and road / not-road labels on one 8 cm pixel grid.

Sources (see US_AREAS in areas.py):
- denver: leaf-off DRAPP photos (7.6 cm, public on S3, NAD83 Colorado Central in US feet), resampled to 8 cm
  in UTM 13N. Labels from DRCOG's regional edge-of-pavement polygons: "Corridor" is road (kerb to kerb, so street
  parking is included), medians are not.
- dc: DC GIS orthophotos, natively 8 cm in Maryland State Plane (EPSG:26985), so used on their own grid. Labels
  from DC's planimetric Roads layer (roads, intersections, alleys) plus crosswalks from the Sidewalks layer;
  islands, parking lots and paved drives are not road.

Road layers map every road, so pixels they don't cover are labelled not road. Pixels outside the source's
coverage (outside the District, or without photos) are IGNORE.

Writes data/<source>/<area>_aerial_<photo year>.npy (source denver or dc; rows x cols x RGB, row 0 = north) and
data/<source>/<area>_classes_8cm.npy (0 not road, 1 road, 255 ignore) with a .json that center_patch.py reads
(class table, pixel counts, the imagery file) and a _preview.png. Photo downloads resume where they stopped; --labels-only redoes the labels.

    python us_fetch.py denver_downtown dc_capitol_hill_2021 ...
"""
import argparse
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")  # S3 has no .aux/.ovr sidecars to look for
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402

from areas import US_AREAS, area_dir  # noqa: E402
from rasterize import IGNORE  # noqa: E402

RES = 0.08
CRS = {"denver": 32613, "dc": 26985}  # UTM 13N; Maryland State Plane (metres), the DC photos' own grid
CLASSES = {0: ("not_road", (0, 0, 0)), 1: ("road", (90, 90, 90))}  # rasterize.py's colours for other / road
WORKERS = 4
RETRIES = 5
DC_TILE = 2500  # px per photo request; the server allows up to 15000
DENVER_STRIP = 1024  # rows per photo strip: DRAPP tiles store whole scanlines, so full-width strips read each once
LABEL_TILE = 4096
PREVIEW_STEP = 16

DRAPP = "https://colorado-public-imagery.s3.us-west-2.amazonaws.com"
DRAPP_CRS = "EPSG:6428"  # NAD83(2011) Colorado Central, US feet
# DRAPP tile names: {N|S}{10-mile band}{E|W}{10-mile column}{mile row from the north}{mile column}{quarter a-d},
# e.g. N1W288c; quarters are 2640 ft (a NW, b NE, c SW, d SE) at 0.25 ft (7.6 cm). Rural areas only have whole-
# mile tiles without the quarter letter (e.g. S1W235) at 0.5 ft (15 cm), upsampled here. Origins read from the
# tiles' world files.
DRAPP_X0, DRAPP_Y0, MILE, BLOCK = 3141392.1248, 1695031.8748, 5280.0, 52800.0
DRCOG_ROADS = "https://gis.drcog.org/server/rest/services/RDC/PLANIMETRICS_{year}_EDGE_PAVEMENT_POLYGON_TOTAL/MapServer/0"
DC_ORTHO = "https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_{year}/ImageServer/exportImage"
DC_PLANIMETRICS = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Planimetrics_{year}/MapServer"
DC_BOUNDARY = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Administrative_Other_Boundaries_WebMercator"
               "/MapServer/10")
DC_ROADS = {"ROAD": 1, "INTERSECTION": 1, "ALLEY": 1,  # HIDDEN ROAD etc. run under bridges and decks: left out
            "PAVED MEDIAN ISLAND": 0, "PAVED TRAFFIC ISLAND": 0, "UNPAVED MEDIAN ISLAND": 0,
            "UNPAVED TRAFFIC ISLAND": 0, "PARKING LOT": 0, "PAVED DRIVE": 0}


def get(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    for attempt in range(RETRIES):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def query(client: httpx.Client, layer: str, bbox: tuple, crs: int, fields: str = "*") -> list[dict]:
    """GeoJSON features of an ArcGIS layer that intersect bbox (in crs), in crs, fetched page by page."""
    features = []
    while True:
        data = get(client, f"{layer}/query", {
            "where": "1=1", "geometry": ",".join(f"{v:.3f}" for v in bbox), "geometryType": "esriGeometryEnvelope",
            "inSR": crs, "outSR": crs, "spatialRel": "esriSpatialRelIntersects", "outFields": fields,
            "orderByFields": "OBJECTID", "resultOffset": len(features), "f": "geojson"}).json()
        if "error" in data:
            raise RuntimeError(f"{layer}: {data['error']}")
        features += data["features"]
        if not (data.get("exceededTransferLimit") or data.get("properties", {}).get("exceededTransferLimit")):
            return features


def code_names(client: httpx.Client, layer: str, field: str) -> dict:
    """Coded-value domain of a field: {code: name}. DC's GeoJSON gives FEATURECODE as the numeric code."""
    fields = get(client, layer, {"f": "json"}).json()["fields"]
    return {c["code"]: c["name"] for f in fields if f["name"] == field for c in f["domain"]["codedValues"]}


def grid(area: str):
    """(crs, transform, height, width) of the area's 8 cm grid. Even metres are on the grid, so DC's photos are
    taken on their own pixels."""
    source, bbox, *_ = US_AREAS[area]
    west, south, east, north = transform_bounds("EPSG:4326", f"EPSG:{CRS[source]}", *bbox)
    west, south = np.floor(west / 2) * 2, np.floor(south / 2) * 2
    width, height = int(np.ceil((east - west) / RES)), int(np.ceil((north - south) / RES))
    return CRS[source], from_origin(west, south + height * RES, RES, RES), height, width


def drapp_tile(x: float, y: float) -> str:
    """Name of the DRAPP quarter tile containing (x, y) in Colorado Central feet."""
    k = int(np.ceil((y - DRAPP_Y0) / BLOCK))  # 10-mile band; N1 is the first above DRAPP_Y0
    top = DRAPP_Y0 + k * BLOCK
    row = int((top - y) // MILE)
    j = int(np.floor((x - DRAPP_X0) / BLOCK))  # 10-mile column; E1 is the first east of DRAPP_X0
    left = DRAPP_X0 + j * BLOCK
    col = int((x - left) // MILE)
    quarter = "abcd"[2 * int((top - row * MILE - y) // (MILE / 2)) + int((x - left - col * MILE) // (MILE / 2))]
    return f"{'N%d' % k if k >= 1 else 'S%d' % (1 - k)}{'E%d' % (j + 1) if j >= 0 else 'W%d' % -j}{row}{col}{quarter}"


def drapp_listing(year: int, data_dir: str) -> set[str]:
    """Names of the DRAPP tiles of a year on S3, cached in data_dir."""
    path = f"{data_dir}/drapp_{year}_tiles.json"
    if os.path.exists(path):
        return set(json.load(open(path)))
    names, token = [], None
    with httpx.Client(timeout=60) as client:
        while True:
            params = {"list-type": 2, "prefix": f"DRAPP/DRAPP{year}/", **({"continuation-token": token} if token else {})}
            text = get(client, f"{DRAPP}/", params).text
            names += [k.split("/")[-1].removesuffix(".tif") for k in re.findall(r"<Key>([^<]+\.tif)</Key>", text)]
            token = (re.findall(r"<NextContinuationToken>([^<]+)<", text) or [None])[0]
            if not token:
                break
    json.dump(sorted(names), open(path, "w"))
    return set(names)


def drapp_strip(area: str, transform, row: int, h: int, width: int, available: set[str]) -> np.ndarray:
    """Rows row:row + h of the area's photo, warped from every DRAPP tile they overlap (average resampling,
    7.6 -> 8 cm). Pixels no tile covers stay 0."""
    crs, year = CRS["denver"], US_AREAS[area][2]
    dst_transform = transform * transform.translation(0, row)
    x0, y1 = dst_transform * (0, 0)
    x1, y0 = dst_transform * (width, h)
    bounds = transform_bounds(f"EPSG:{crs}", DRAPP_CRS, x0, y0, x1, y1)
    xs = np.append(np.arange(bounds[0], bounds[2], MILE / 2), bounds[2])  # a point in every quarter tile
    ys = np.append(np.arange(bounds[1], bounds[3], MILE / 2), bounds[3])
    quarters = {drapp_tile(x, y) for x in xs for y in ys}
    # a whole-mile tile where its quarters don't exist; drawn first, so any quarter tiles go on top
    names = sorted({q[:-1] for q in quarters} & available) + sorted(quarters & available)
    out, covered = np.zeros((3, h, width), np.uint8), np.zeros((h, width), bool)
    for name in names:
        with rasterio.open(f"/vsicurl/{DRAPP}/DRAPP/DRAPP{year}/{name}.tif") as src:
            pad = 4 * abs(src.transform.a)  # a few source pixels, so the averaging at the strip edge has its inputs
            window = from_bounds(bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad, src.transform)
            try:
                window = window.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError:
                continue
            data = src.read([1, 2, 3], window=window)
            src_transform = src.window_transform(window)
            warped, seen = np.zeros((3, h, width), np.uint8), np.zeros((h, width), np.uint8)
            reproject(data, warped, src_transform=src_transform, src_crs=src.crs, dst_transform=dst_transform,
                      dst_crs=f"EPSG:{crs}",  # averaging down from 7.6 cm, interpolating up from 15 cm
                      resampling=Resampling.average if abs(src.transform.a) < 0.3 else Resampling.bilinear)
            reproject(np.ones(data.shape[1:], np.uint8), seen, src_transform=src_transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=f"EPSG:{crs}", resampling=Resampling.nearest)
            seen = seen.astype(bool)
            out[:, seen] = warped[:, seen]
            covered |= seen
    image = out.transpose(1, 2, 0)
    image[covered] = np.maximum(image[covered], 1)  # so (0, 0, 0) only ever means "no photo"
    return image


def dc_tile(client: httpx.Client, area: str, transform, row: int, col: int, h: int, w: int) -> np.ndarray:
    x0, y1 = transform * (col, row)
    response = get(client, DC_ORTHO.format(year=US_AREAS[area][2]), {
        "bbox": f"{x0:.3f},{y1 - h * RES:.3f},{x0 + w * RES:.3f},{y1:.3f}", "bboxSR": CRS["dc"],
        "imageSR": CRS["dc"], "size": f"{w},{h}", "format": "jpg", "compressionQuality": 90, "bandIds": "0,1,2",
        "interpolation": "RSP_NearestNeighbor", "f": "image"})
    if not response.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"ImageServer returned {response.headers.get('content-type')}: {response.text[:200]}")
    image = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
    if image.shape != (h, w, 3):
        raise ValueError(f"expected {h}x{w}, got {image.shape}")
    return np.maximum(image, 1)  # so (0, 0, 0) only ever means "no photo"


def fetch_imagery(area: str, path: str, data_dir: str):
    source = US_AREAS[area][0]
    crs, transform, height, width = grid(area)
    if source == "denver":
        chunks = [(row, 0, min(DENVER_STRIP, height - row), width) for row in range(0, height, DENVER_STRIP)]
        available = drapp_listing(US_AREAS[area][2], data_dir)
    else:
        chunks = [(row, col, min(DC_TILE, height - row), min(DC_TILE, width - col))
                  for row in range(0, height, DC_TILE) for col in range(0, width, DC_TILE)]
    progress_path = f"{path.removesuffix('.npy')}.progress.json"
    try:
        with open(progress_path) as file:
            done = {tuple(c) for c in json.load(file)}
        out = np.load(path, mmap_mode="r+")
        assert out.shape == (height, width, 3)
    except (FileNotFoundError, AssertionError):
        done = set()
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(height, width, 3))
    todo = [c for c in chunks if c[:2] not in done]
    print(f"{area}: {width} x {height} px ({out.nbytes / 1e9:.1f} GB), {len(chunks)} chunks, {len(todo)} to fetch")
    client = httpx.Client(timeout=300)

    def work(chunk):
        row, col, h, w = chunk
        if source == "denver":
            return drapp_strip(area, transform, row, h, w, available)
        return dc_tile(client, area, transform, row, col, h, w)

    start = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, (chunk, image) in enumerate(zip(todo, pool.map(work, todo)), start=1):
            row, col, h, w = chunk
            out[row:row + h, col:col + w] = image
            done.add((row, col))
            if i % 10 == 0 or i == len(todo):
                out.flush()
                with open(progress_path, "w") as file:
                    json.dump(sorted(done), file)
                print(f"  {i}/{len(todo)} chunks, {time.time() - start:.0f}s", end="\r", flush=True)
    del out
    os.remove(progress_path)
    print()


def label_shapes(area: str, bbox: tuple, crs: int) -> tuple[list, list]:
    """([(geometry, class id)] in burn order, later wins; [coverage geometries] or [] for everywhere)."""
    source, _, _, year = US_AREAS[area]
    client = httpx.Client(timeout=300)
    if source == "denver":
        # Fetched in their own CRS (the photos' too) and projected here like the photos: the server's
        # NAD83 -> WGS84 step differs from PROJ's, which put the labels ~1.1 m north-west of the photos
        native = transform_bounds(f"EPSG:{crs}", DRAPP_CRS, *bbox)
        features = query(client, DRCOG_ROADS.format(year=year), native, int(DRAPP_CRS.split(":")[1]),
                         fields="OBJECTID,Type")
        types = [f["properties"]["Type"] for f in features]
        print(f"  DRCOG {year}: " + ", ".join(f"{t} {types.count(t)}" for t in sorted(set(types), key=str)))
        geoms = [transform_geom(DRAPP_CRS, f"EPSG:{crs}", f["geometry"]) for f in features]
        shapes = [(g, 1) for g, t in zip(geoms, types) if t == "Corridor"]
        shapes += [(g, 0) for g, t in zip(geoms, types) if t == "Median"]  # medians sit on corridors
        return shapes, []
    base = DC_PLANIMETRICS.format(year=year)
    shapes = []
    for layer, mapping in ((10, DC_ROADS), (11, {"CROSSWALK": 1})):  # Roads; crosswalks from Sidewalks, on top
        names = code_names(client, f"{base}/{layer}", "FEATURECODE")
        features = query(client, f"{base}/{layer}", bbox, crs, fields="OBJECTID,FEATURECODE")
        codes = [names.get(f["properties"]["FEATURECODE"]) for f in features]
        print(f"  DC {year} layer {layer}: " + ", ".join(f"{c} {codes.count(c)}" for c in sorted(set(codes), key=str)))
        kept = [(f["geometry"], mapping[c]) for f, c in zip(features, codes) if c in mapping]
        shapes += sorted(kept, key=lambda s: s[1] == 0) if layer == 10 else kept  # islands etc. on top of roads
    boundary = [f["geometry"] for f in query(client, DC_BOUNDARY, bbox, crs, fields="OBJECTID")]
    return shapes, boundary


def write_labels(area: str, data_dir: str):
    source, _, photo_year, label_year = US_AREAS[area]
    crs, transform, height, width = grid(area)
    imagery_file = f"{area}_aerial_{photo_year}.npy"
    imagery = np.load(f"{data_dir}/{imagery_file}", mmap_mode="r")
    x0, y1 = transform * (0, 0)
    x1, y0 = transform * (width, height)
    shapes, coverage = label_shapes(area, (x0, y0, x1, y1), crs)
    geoms = np.array([shapely.geometry.shape(g) for g, _ in shapes], dtype=object)
    values = np.array([v for _, v in shapes], dtype=np.uint8)
    coverage = shapely.union_all([shapely.geometry.shape(g) for g in coverage]) if coverage else None
    tree = shapely.STRtree(geoms)

    out_path = f"{data_dir}/{area}_classes_8cm"
    out = np.lib.format.open_memmap(f"{out_path}.npy", mode="w+", dtype=np.uint8, shape=(height, width))
    preview = np.zeros((-(-height // PREVIEW_STEP), -(-width // PREVIEW_STEP)), dtype=np.uint8)
    counts = np.zeros(256, dtype=np.int64)
    for row in range(0, height, LABEL_TILE):
        for col in range(0, width, LABEL_TILE):
            h, w = min(LABEL_TILE, height - row), min(LABEL_TILE, width - col)
            tile_transform = transform * transform.translation(col, row)
            tx0, ty1 = tile_transform * (0, 0)
            tx1, ty0 = tile_transform * (w, h)
            hits = np.sort(tree.query(shapely.box(tx0, ty0, tx1, ty1)))  # keep burn order
            tile = (rasterize(zip(geoms[hits], values[hits]), out_shape=(h, w), transform=tile_transform, fill=0,
                              dtype=np.uint8) if len(hits) else np.zeros((h, w), np.uint8))
            if coverage is not None:
                inside = rasterize([(coverage, 1)], out_shape=(h, w), transform=tile_transform, fill=0, dtype=np.uint8)
                tile[inside == 0] = IGNORE
            tile[(np.asarray(imagery[row:row + h, col:col + w]) == 0).all(axis=2)] = IGNORE  # no photo
            out[row:row + h, col:col + w] = tile
            counts += np.bincount(tile.ravel(), minlength=256)
            small = tile[::PREVIEW_STEP, ::PREVIEW_STEP]
            preview[row // PREVIEW_STEP:row // PREVIEW_STEP + small.shape[0],
                    col // PREVIEW_STEP:col // PREVIEW_STEP + small.shape[1]] = small
        out.flush()
    del out
    meta = {
        "crs": f"EPSG:{crs}",
        "resolution_m": RES,
        "origin_top_left": [x0, y1],
        "shape": [height, width],
        "transform": list(transform)[:6],
        "classes": {i: name for i, (name, _) in CLASSES.items()},
        "ignore": IGNORE,
        "imagery": imagery_file,
        "source": source,
        "photo_year": photo_year,
        "label_year": label_year,
        "pixel_counts": {**{name: int(counts[i]) for i, (name, _) in CLASSES.items()}, "ignore": int(counts[IGNORE])},
    }
    with open(f"{out_path}.json", "w") as file:
        json.dump(meta, file, indent=2)
    palette = np.full((256, 3), 255, dtype=np.uint8)  # IGNORE previews as white
    palette[:len(CLASSES)] = [rgb for _, rgb in CLASSES.values()]
    Image.fromarray(palette[preview]).save(f"{out_path}_preview.png")
    total = max(sum(meta["pixel_counts"].values()), 1)
    print(f"  labels: " + ", ".join(f"{k} {v / total:.1%}" for k, v in meta["pixel_counts"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("areas", nargs="+", choices=sorted(US_AREAS), metavar="AREA", help="areas from US_AREAS")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--labels-only", action="store_true", help="only redo the labels of already fetched photos")
    args = parser.parse_args()
    for area in args.areas:
        folder = area_dir(area, args.data_dir)
        path = f"{folder}/{area}_aerial_{US_AREAS[area][2]}.npy"
        if not args.labels_only and (not os.path.exists(path) or os.path.exists(path.removesuffix(".npy") + ".progress.json")):
            fetch_imagery(area, path, folder)
        print(f"{area}: labels")
        write_labels(area, folder)


if __name__ == "__main__":
    main()
