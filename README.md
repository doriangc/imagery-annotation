# bgt-fit

Training data and models for predicting road cross-section classes (road, bike lane, sidewalk, parking,
landscaping, ...) from 8 cm aerial imagery, with BGT as labels. See [ROADMAP.md](ROADMAP.md) for goals and status.

## Setup on a fresh machine (e.g. a GPU box)

```sh
git clone <this repo> && cd bgt-fit
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # see the comment in it for CPU-only machines

python bgt_fetch.py                    # BGT for Houten        -> data/houten_bgt.pkl          (a few min)
python rasterize.py                    # 8 cm class raster     -> data/houten_classes_8cm.*    (~1 min, 2.7 GB)
python aerial_imagery_fetch.py         # 2026 aerial photos    -> data/houten_aerial_2026.*    (~3 min, 8.1 GB)

python center_patch.py                 # train on imagery; uses the GPU when available
```

`data/` is not in git: regenerating it is faster than uploading ~11 GB. On Colab, run the commands with `!`
in a notebook and copy checkpoints (`data/*.pt`) to Google Drive, since the disk is wiped when the session ends.

Useful `center_patch.py` options: `--model unet` (with `--encoder resnet34` and `--encoder-weights imagenet|none`),
`--batch` (raise it on a GPU), `--workers` (window-sampling processes),
`--steps`, `--eval-every`, `--no-augment`, and `--input labels` for the labels-only test with its copy-left baseline.
