# bgt-fit

Training data and models for predicting road cross-section classes (road, bike lane, sidewalk, parking,
landscaping, ...) from 8 cm aerial imagery, with BGT as labels. See [ROADMAP.md](ROADMAP.md) for goals and status.

## Setup on a fresh machine (e.g. a GPU box)

```sh
git clone <this repo> && cd bgt-fit
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # see the comment in it for CPU-only machines

AREAS="houten utrecht leidsche_rijn amersfoort amsterdam nijmegen veere"   # see areas.py
python bgt_fetch.py $AREAS                                       # BGT -> data/nl/<area>_bgt.pkl
for a in $AREAS; do python rasterize.py $a; done                 # labels -> data/nl/<area>_classes_8cm.*
for a in $AREAS; do python aerial_imagery_fetch.py $a; done      # photos -> data/nl/<area>_aerial_2026.*

python center_patch.py --model unet --areas houten utrecht leidsche_rijn amsterdam nijmegen veere \
    --test-areas amersfoort                                      # uses the GPU when available
```

Each 3 × 3 km area is about 1.4 GB of labels and 4.3 GB of imagery; Houten (4.2 × 4.1 km) is 2.7 + 8.1 GB.
`rasterize.py` marks road polygons under OpenStreetMap fietsstraten as ignored (255), since municipalities label
them inconsistently; `python label_report.py` writes `results/label_consistency.md` comparing BGT conventions.

Area files live in a folder per region, `data/nl/`, `data/denver/` and `data/dc/` (`area_dir` in `areas.py`), and
checkpoints in `data/`. `python move_data.py` (`--dry-run` to preview) moves files fetched before this layout.

`data/` is not in git: regenerating it is faster than uploading ~11 GB. On Colab, run the commands with `!`
in a notebook and copy checkpoints (`data/*.pt`) to Google Drive, since the disk is wiped when the session ends.

## US road-only model

Road (kerb to kerb, so street parking, plus alleys and crosswalks) vs not road, for the Denver region and
Washington, DC (`US_AREAS` in `areas.py`). `us_fetch.py AREA ...` writes the same files as the NL scripts, in `data/denver/` and `data/dc/`:
- Denver: leaf-off DRAPP 2022 photos from the public `colorado-public-imagery` S3 bucket (7.6 cm; 15 cm in rural
  areas such as Genesee), warped to 8 cm UTM 13N, with DRCOG's 2024 edge-of-pavement polygons.
- DC: DC GIS orthophotos on their native 8 cm grid, with that year's planimetrics (2025 photos use the 2023
  planimetrics); outside the District is ignored.

`center_patch.py` takes the class table from the label `.json`, so it trains the two-class model with every option
above; US and NL areas can't be mixed in one run. `./us_road.sh` fetches all 25 areas (~142 GB) and trains.

Useful `center_patch.py` options: `--areas` / `--test-areas`, `--size 384`,
`--class-weight-power 0.25`, `--evaluate CHECKPOINT`, `--model unet` (with `--encoder resnet34` and `--encoder-weights imagenet|none`),
`--batch` (raise it on a GPU), `--workers` (window-sampling processes),
`--steps`, `--eval-every`, `--no-augment`, and `--input labels` for the labels-only test with its copy-left baseline.
