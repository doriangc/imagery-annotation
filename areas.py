"""Named study areas as (west, south, east, north) in lon/lat. Data files are named after the area and kept in a
folder per region (area_dir): data/nl/<name>_bgt.pkl, data/nl/<name>_classes_8cm.*, data/nl/<name>_aerial_2026.*
(US_AREAS: see below). Checkpoints stay in data/."""
import os

AREAS = {
    "houten": (5.1400516, 52.008016, 5.20152, 52.0446192),  # ~4.2 x 4.1 km, the original area
    # ~3 x 3 km each
    "utrecht": (5.100, 52.0765, 5.144, 52.1035),  # centre and east; kept apart from Leidsche Rijn
    "leidsche_rijn": (5.033, 52.0815, 5.077, 52.1085),
    "amersfoort": (5.365, 52.1425, 5.409, 52.1695),
    "amsterdam": (4.878, 52.3415, 4.922, 52.3685),  # De Pijp, Rivierenbuurt, part of Oost
    "nijmegen": (5.831, 51.829, 5.875, 51.856),
    "veere": (3.628, 51.532, 3.672, 51.559),  # Veere town (east edge), Gapinge, countryside; less Veerse Meer
    # boulevard towns: wide multi-lane streets with medians, like Amersfoort's avenue (r22689_c8230)
    "den_haag": (4.276, 52.070, 4.320, 52.097),  # Koningskade, Zeestraat, east end of Laan van Meerdervoort
    "rotterdam": (4.455, 51.907, 4.499, 51.934),  # Coolsingel, Weena, Westersingel, Blaak
    "eindhoven": (5.465, 51.440, 5.509, 51.467),  # Kennedylaan, Fuutlaan, ring road
    # second test town, unlike Amersfoort: northern, compact old centre inside the diepen, heavy cycling network
    "groningen": (6.545, 53.204, 6.589, 53.231),
}

# US areas for the road-only model (us_fetch.py), ~3 x 3 km each, as (source, bbox, photo year, label year).
# Labels are road (carriageway incl. street parking, alleys, crosswalks) vs not road. Files are
# data/<source>/<name>_classes_8cm.* and data/<source>/<name>_aerial_<photo year>.npy.
US_AREAS = {
    # Denver region: leaf-off DRAPP photos (7.6 cm, public on S3) + DRCOG regional edge-of-pavement polygons
    "denver_downtown": ("denver", (-105.005, 39.730, -104.970, 39.757), 2022, 2024),  # CBD, Capitol Hill, Five Points
    "denver_highlands": ("denver", (-105.050, 39.750, -105.015, 39.777), 2022, 2024),  # older grid with alleys
    "denver_wash_park": ("denver", (-104.990, 39.690, -104.955, 39.717), 2022, 2024),
    "denver_park_hill": ("denver", (-104.950, 39.730, -104.915, 39.757), 2022, 2024),
    "lakewood": ("denver", (-105.100, 39.700, -105.065, 39.727), 2022, 2024),  # Belmar, suburban arterials
    "aurora": ("denver", (-104.850, 39.680, -104.815, 39.707), 2022, 2024),
    "littleton": ("denver", (-105.030, 39.600, -104.995, 39.627), 2022, 2024),
    "arvada": ("denver", (-105.100, 39.790, -105.065, 39.817), 2022, 2024),
    "highlands_ranch": ("denver", (-104.9975, 39.5315, -104.9625, 39.5585), 2022, 2024),  # curving suburban streets
    # rural: plains town grid, farmland, gravel county roads, I-70; kept west of the 2022 building sites in the east
    "bennett": ("denver", (-104.4545, 39.7465, -104.4195, 39.7735), 2022, 2024),
    "golden": ("denver", (-105.240, 39.740, -105.205, 39.767), 2022, 2024),  # test: small town, foothills
    "boulder": ("denver", (-105.290, 40.000, -105.255, 40.027), 2022, 2024),  # test: a separate city
    # test: mountain rural (winding roads, pines); DRAPP only has 15 cm photos here, upsampled to 8 cm
    "genesee": ("denver", (-105.2925, 39.6735, -105.2575, 39.7005), 2022, 2024),
}
# DC: orthophotos and planimetrics from DC GIS, in the leaf-off years only (2019 and 2023 are flown in summer
# canopy). There are no 2025 planimetrics, so the 2025 photos get the 2023 ones. Both years of an area share its
# box, so they also share its validation strip.
for _name, _bbox in {
    "dc_capitol_hill": (-77.005, 38.875, -76.970, 38.902),  # rowhouses, alleys, H St NE
    "dc_downtown": (-77.050, 38.895, -77.015, 38.922),  # downtown, Dupont, Logan Circle
    "dc_columbia_heights": (-77.045, 38.922, -77.010, 38.949),  # Columbia Heights, Petworth
    "dc_northwest": (-77.090, 38.925, -77.055, 38.952),  # Cleveland Park, Tenleytown; detached houses
    "dc_anacostia": (-77.000, 38.840, -76.965, 38.867),  # east of the river, hills
    "dc_brookland": (-77.000, 38.920, -76.965, 38.947),  # test
}.items():
    for _photo, _labels in ((2021, 2021), (2025, 2023)):
        US_AREAS[f"{_name}_{_photo}"] = ("dc", _bbox, _photo, _labels)


def area_dir(area: str, data_dir: str = "data") -> str:
    """The folder of an area's files, data_dir/<region> (nl, denver or dc), created if missing."""
    path = os.path.join(data_dir, US_AREAS[area][0] if area in US_AREAS else "nl")
    os.makedirs(path, exist_ok=True)
    return path
