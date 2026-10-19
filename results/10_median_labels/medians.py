"""How do towns label paved strips that run between vehicle surfaces (medians)?

Candidate: a paved polygon (wegdeel or ondersteunendwegdeel, not itself road/parking/bike lane) that is long and
narrow (width ~ 2*area/perimeter between 1.5 and 10 m, perimeter >= 60 m) and whose boundary touches road,
parking or bike lane for at least 65% of its length, i.e. on both long sides (a sidewalk touches on one side).
"""
import json, pickle, sys, collections
import numpy as np, pyproj, shapely
sys.path.insert(0, "/home/dorian/bgt-fit")
from areas import AREAS
from rasterize import classify, PAVED

to_rd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
project = lambda g: shapely.transform(g, lambda xy: np.column_stack(to_rd.transform(xy[:, 0], xy[:, 1])))
VEHICLE = {"road", "parking", "bike_lane", "speed_bump"}
results = {}
for area in sys.argv[1:]:
    data = pickle.load(open(f"/home/dorian/bgt-fit/data/{area}_bgt.pkl", "rb"))
    rows = [(c, f["properties"]) for c in ("wegdeel", "ondersteunendwegdeel") for f in data[c]]
    geoms = project(shapely.from_geojson([json.dumps(f["geometry"]) for c in ("wegdeel", "ondersteunendwegdeel") for f in data[c]]))
    geoms = shapely.make_valid(geoms)
    bbox = shapely.box(*to_rd.transform_bounds(*AREAS[area]))
    cls = np.array([classify(c, p) or "-" for c, p in rows])
    vehicle = np.isin(cls, list(VEHICLE))
    tree = shapely.STRtree(geoms[vehicle])
    vgeoms = geoms[vehicle]
    tally, examples = collections.Counter(), collections.defaultdict(list)
    for i, ((coll, p), g) in enumerate(zip(rows, geoms)):
        if vehicle[i] or p["fysiek_voorkomen"] not in PAVED or not g.intersects(bbox):
            continue
        perim = g.length
        if perim < 60:
            continue
        width = 2 * g.area / perim
        if not 1.5 <= width <= 10:
            continue
        near = vgeoms[tree.query(g, predicate="dwithin", distance=0.5)]
        if not len(near):
            continue
        touch = shapely.intersection(g.boundary, shapely.union_all(near).buffer(0.5)).length / perim
        if touch < 0.65:
            continue
        key = f"{coll}: {p['functie']} -> {cls[i]}"
        tally[key] += g.area
        c = g.representative_point()
        examples[key].append((round(g.area), round(c.x), round(c.y)))
    total = sum(tally.values())
    results[area] = (total, tally, dict(examples))
    print(f"\n{area}: {total:,.0f} m2 of median-like paved strips")
    for k, v in tally.most_common():
        if v / total >= 0.02:
            print(f"  {v / total:5.0%}  {k}   e.g. {sorted(examples[k], reverse=True)[:2]}")
json.dump({a: [t, dict(c), e] for a, (t, c, e) in results.items()},
          open("medians.json", "w"))
