import argparse
import httpx
import pickle

from areas import AREAS, area_dir

PAGE_SIZE = 1000
API = "https://api.pdok.nl/lv/bgt/ogc/v1/collections/{collection}/items"
COLLECTIONS = ("wegdeel", "ondersteunendwegdeel", "begroeidterreindeel")
KEEP_PROPERTIES = (
    "lokaal_id", "functie", "fysiek_voorkomen", "plus_functie", "plus_fysiek_voorkomen",
    "tijdstip_registratie", "relatieve_hoogteligging",
)

def fetch(bbox: tuple) -> dict:
    """All current BGT objects of COLLECTIONS around `bbox` (west, south, east, north),
    as {collection: [GeoJSON features in lon/lat]} with a reduced set of properties."""
    west, south, east, north = bbox
    box = f"{west},{south},{east},{north}"
    client = httpx.Client(timeout=120, headers={"User-Agent": "road-centerline-annotator/0.1"})
    out: dict[str, list] = {}
    try:
        for collection in COLLECTIONS:
            url = API.format(collection=collection)
            params = {"f": "json", "bbox": box, "limit": PAGE_SIZE}
            features = []
            while url:
                response = client.get(url, params=params)
                response.raise_for_status()
                page = response.json()
                for f in page.get("features", []):
                    props = f.get("properties", {})
                    if props.get("eind_registratie") or props.get("termination_date"):
                        continue  # no longer current
                    features.append({
                        "type": "Feature",
                        "geometry": f["geometry"],
                        "properties": {k: props.get(k) for k in KEEP_PROPERTIES},
                    })
                url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
                params = None  # the next link carries them
            out[collection] = features
    finally:
        client.close()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch BGT for named areas (see areas.py) to data/nl/<area>_bgt.pkl")
    parser.add_argument("areas", nargs="*", default=["houten"], choices=sorted(AREAS), metavar="AREA",
                        help=f"one or more of: {', '.join(AREAS)}")
    args = parser.parse_args()

    for area in args.areas:
        out = fetch(AREAS[area])
        with open(f"{area_dir(area)}/{area}_bgt.pkl", "wb") as file:
            pickle.dump(out, file)
        print(f"{area}: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()), flush=True)
