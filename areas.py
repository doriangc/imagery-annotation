"""Named study areas as (west, south, east, north) in lon/lat. Data files are named after the area:
data/<name>_bgt.pkl, data/<name>_classes_8cm.*, data/<name>_aerial_2026.*"""

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
}
