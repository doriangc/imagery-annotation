# Results

Images kept for reference. Grid coordinates (`r…_c…`) are the top-left pixel (row, column) in the 8 cm
raster `data/houten_classes_8cm.npy`. Rerun a block with `python visualize.py <checkpoint> --at ROW COL`.

## 01_labels: BGT class rasters

Colour previews of the rasterized BGT labels for the Houten bbox, at 2 m per preview pixel.

- `houten_classes_10cm.png`: the first 10 cm raster.
- `houten_classes_8cm.png`: the 8 cm raster on the aerial photos' grid, used for all training.

Black is "other" (buildings, water, yards, bare ground, ...). The legend is in `rasterize.py` `CLASSES`.

## 02_imagery_alignment: aerial photos vs labels

- `houten_aerial_2026_overview.jpg`: the whole `2026_orthoHR` download at 3.2 m per pixel.
- `overlay_*.png`: 32 × 32 m crops, photo on the left and BGT boundaries in magenta on the right. They show:
  - road and path edges match to within about a pixel
  - roofs are offset from the BGT ground outline, because buildings lean in aerial photos
  - paths under tree canopy are labelled but invisible

## 03_unet_imagenet_10k, 04_unet_scratch_10k, 05_unet_imagenet_50k: predictions on validation blocks

U-Net with a ResNet-34 encoder on 256 px windows, batch 32:

- 03: ImageNet-pretrained, 10,000 steps
- 04: trained from scratch, 10,000 steps
- 05: ImageNet-pretrained, 50,000 steps

All blocks are 61 × 61 m from the southern validation strip, which the models never trained on.

Panels: photo | BGT labels | prediction | errors in red.

| Block | Contents | 03 ImageNet 10k | 04 scratch 10k | 05 ImageNet 50k |
|---|---|---|---|---|
| `r44674_c42357` | parking area, paved plaza with tree planters | 86.5% | 76.1% | 90.3% |
| `r46896_c31611` | residential street, parking bays, driveways | 87.9% | 80.4% | 93.3% |
| `r43637_c28222` | residential street, bike lane, footpath between back gardens | 91.4% | 86.1% | 94.8% |

Overall validation mIoU: 0.46 (03), 0.34 (04), 0.51 (05).

In 05, compared with 03:

- The footpath between back gardens is found along most of its length, with a gap in shade.
- Driveways appear for the first time (45% of driveway pixels in `r43637_c28222`, but 0% in `r46896_c31611`).
- Sidewalk goes from 33–59% to about 80% of its pixels across the blocks.
- Edges are much straighter.

Typical failures in the 10k ImageNet model (03):

- narrow footpaths are missed
- boundaries are blobby
- driveways are never predicted
- paved plazas are confused with sidewalk, other and parking
