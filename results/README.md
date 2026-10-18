# Results

Images kept for reference. Grid coordinates (`r…_c…`) are the top-left pixel (row, column) in the 8 cm
raster `data/houten_classes_8cm.npy`. Rerun a block with `python visualize.py <checkpoint> --at ROW COL`.

## Reference blocks

Fixed Amersfoort (test town) blocks, rendered with every new model so changes can be compared side by side:

| Block | Why |
|---|---|
| `r22689_c8230` | **hard case**: wide avenue with paved tree medians, bus lane, bicycle parking, tree and building shadows |
| `r28160_c16065` | junction with traffic islands, red bike paths, level crossing |
| `r18804_c10012` | residential street: parking lanes, speed bump, alley between back gardens |
| `r21686_c31355` | urban woodland next to a street (the farmland relabel) |

```
python visualize.py CHECKPOINT --area amersfoort --stride 64 --tta \
    --at 22689 8230 --at 28160 16065 --at 18804 10012 --at 21686 31355 --out results/NN_name/vis
```

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

## 06_unet_multi: multi-town model on Amersfoort (held-out test town)

U-Net ResNet-34 (ImageNet), 384 px windows, 128 px target, class weight power 0.25, 50,000 steps on six towns
(see ROADMAP). Grid coordinates refer to `data/amersfoort_classes_8cm.npy`. Test mIoU 0.479.

| Block | Contents | Accuracy |
|---|---|---|
| `r18804_c10012` | residential street with parking lanes, speed bump, alley between back gardens | 93.0% |
| `r2877_c733` | random street block | 94.7% |
| `r22689_c8230` | wide avenue with paved tree medians (BGT: traffic island), bicycle parking | 65.1% |
| `r28160_c16065` | junction with traffic islands, red bike paths, level crossing | 77.6% |
| `r21686_c31355` | urban woodland (BGT `loofbos` → agri_natural) next to a street | 11.6% |
| `r32689_c31245` | park (BGT groenvoorziening) | 98.7% |

- Streets, parking lanes, speed bumps and red bike paths are found well; errors there are mostly 1–3 px edges.
- The narrow alley between back gardens is still missed.
- Agri/natural scoring 0.02 on Amersfoort is a class-definition problem: training towns' agri_natural is
  farmland (Houten, Veere), while Amersfoort's is mostly urban woodland, which the model calls landscaping.
- Amersfoort maps wide paved, tree-lined medians as traffic island; the model splits them between sidewalk,
  road and island. A convention question more than a model failure.

`vis_overlap_tta_*`: the same model and blocks predicted with `--stride 64 --tta` (overlapping windows averaged,
plus the 8 rotations/flips). The straight-edged seam rectangles disappear, e.g. the "other" patches in the
woodland block; the patchy predictions on the avenue medians stay, since they come from the model's uncertainty.
These were rendered after the farmland relabel, so the woodland is landscaping in the BGT panel.

## 07_lighting_augmentation: what `--lighting` does to training windows

`examples.png`: three 384 px training-size windows (Amersfoort avenue, Amersfoort residential street, Houten),
the original on the left and five random draws of `photometric()` in `center_patch.py`. Each draw varies
brightness, contrast, gamma, saturation and colour cast; about half add a soft-edged synthetic shadow. Labels
are unchanged.

## 08_unet_multi_v2: farmland and paved-island labels, lighting augmentation

Same setup as 06 plus `--lighting`, trained on the relabelled rasters (class 11 farmland only, planted traffic
islands as landscaping). Reference blocks rendered with `--stride 64 --tta`. Test mIoU 0.544 (0.545 with TTA).

| Block | 06 (overlap+TTA) | 08 (overlap+TTA) |
|---|---|---|
| `r22689_c8230` avenue | 72.8% | 76.5% |
| `r28160_c16065` junction | - | 77.2% |
| `r18804_c10012` residential | 93.1% | 92.8% |
| `r21686_c31355` woodland | 88.6% | 88.9% |

- Avenue: the bicycle parking is now sidewalk (sidewalk 80% → 89% of its pixels), but the paved medians are
  mostly predicted as road; traffic island drops to 16% of its pixels. Parking on the north side is partly
  predicted as bike lane.
- The other blocks are essentially unchanged.
