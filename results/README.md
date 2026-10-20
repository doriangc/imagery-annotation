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

## 09_unet_context: with a 4x context view (`--context-scale 4`)

Same as 08 plus the 123 m context view. Test mIoU 0.549 (08: 0.544). Reference blocks with `--stride 64 --tta`:

| Block | 08 | 09 |
|---|---|---|
| `r22689_c8230` avenue | 76.5% | 61.1% |
| `r28160_c16065` junction | 77.2% | 77.8% |
| `r18804_c10012` residential | 92.8% | 93.4% |
| `r21686_c31355` woodland | 88.9% | 76.2% |

- Avenue: the medians are now mostly sidewalk (island 6% of its pixels), and the bicycle parking is "other" again.
- Woodland: large straight-edged patches of "other" that follow the window grid, even with overlapping windows:
  neighbouring windows disagree strongly, which suggests the context view sways whole windows at once.

## 10_median_labels: how training towns label median-like paved strips

Question: do training towns label paved medians like Amersfoort's avenue (`r22689_c8230`) as something other than
traffic island? `medians.py` finds paved, long, narrow polygons whose boundary touches road, parking or bike lane
for at least 65% of its length. `random_strips_utrecht_houten.png` shows random matches (photo | BGT).

- The automatic count is not reliable: most matches are ordinary footpaths between a road and a parking lane or
  bike path, station platforms or car-park walkways, all correctly sidewalk.
- Real medians in Houten (arterial roads) and Utrecht (road/bike path separators) are labelled traffic island,
  so there's no sign of a convention difference for medians as such.
- Amersfoort's medians are unusual: wide, tree-lined, with tree pits, between a carriageway and a parking lane.
  In the training towns that look (trees in pits next to parking) almost always belongs to a sidewalk.

## 11_unet_convnext_tiny, 12_unet_convnext_small: ConvNeXt encoders

Same as 08 with `--encoder tu-convnext_tiny` / `tu-convnext_small`. Test mIoU 0.550 / 0.546 (08: 0.544).
Reference blocks with `--stride 64 --tta`:

| Block | 08 (ResNet-34) | 11 (ConvNeXt-T) | 12 (ConvNeXt-S) |
|---|---|---|---|
| `r22689_c8230` avenue | 76.5% | 60.8% | 70.0% |
| `r28160_c16065` junction | 77.2% | 83.4% | 82.4% |
| `r18804_c10012` residential | 92.8% | 92.8% | 93.6% |
| `r21686_c31355` woodland | 88.9% | 94.2% | 90.2% |

- Avenue: medians are sidewalk and road (island 13% / 9% of its pixels); the lane structure is cleaner, but the
  bicycle parking gets a straight-edged rectangle of "other", the same window artefact as in 09.
- Junction and woodland improve, so the reference blocks don't point one way; the test mIoU is level with 08.

## 13_unet_boulevards: three boulevard towns, traffic islands merged into sidewalk

Same as 08 (ResNet-34), trained on nine towns (+ `den_haag`, `rotterdam`, `eindhoven`) with paved traffic islands
labelled sidewalk. Test mIoU 0.559 (08 scored the same way, `--merge traffic_island=sidewalk`: 0.548).
`vis_*` is this model, `v2_merged_*` is 08 rendered with `--merge traffic_island=sidewalk` on the new labels, so
both are scored on the same truth. Reference blocks with `--stride 64 --tta`:

| Block | 08 merged | 13 |
|---|---|---|
| `r22689_c8230` avenue | 82.7% | 80.0% |
| `r28160_c16065` junction | 88.1% | 89.2% |
| `r18804_c10012` residential | 92.8% | 93.2% |
| `r21686_c31355` woodland | 88.9% | 93.5% |

- Avenue: the carriageways are almost all road now (road 89% → 97% of its pixels), and the upper median is a
  clean sidewalk strip. The lower median, between the second carriageway and the parking lane, is still mostly
  road, the bicycle parking gets a straight-edged block of "other" (as in 09, 11 and 12), and tree crowns over
  the lower sidewalk become landscaping.
- `new_labels_*`: the same model against Amersfoort re-rasterized with the fietsstraat fix and level crossings
  as rail (both after this run). The junction's main road is now scored (road 86% → 93%), and the track through
  the junction is rail, which this model, trained with it as road, gets 41% of.

## 14_unet_512: 512 px windows

Same as 13, with 512 px training windows (41 m) instead of 384 px (31 m); same 128 px target. Test mIoU 0.565
(13: 0.559). Rendered against Amersfoort with the fietsstraat fix and level crossings as rail, so compare with
`13_unet_boulevards/new_labels_*`:

| Block | 13 | 14 |
|---|---|---|
| `r22689_c8230` avenue | 80.0% | 81.6% |
| `r28160_c16065` junction | 88.9% | 89.7% |
| `r18804_c10012` residential | 93.2% | 93.1% |
| `r21686_c31355` woodland | 93.5% | 93.5% |

- Avenue: the straight-edged "other" block on the bicycle parking shrinks to the top-left corner, and the
  bicycle parking is mostly sidewalk. The lower median is still road; tree crowns over the lower sidewalk are
  less often landscaping.
- Junction: the track through the crossing is still mostly road (rail 43%); this model was trained with it as road.

## 15_unet_new_labels: retrained on the fietsstraat fix and level crossings as rail

Same as 14, on all ten towns re-rasterized with the new label rules. Test mIoU 0.563 (14 on the same labels:
0.566). Reference blocks with `--stride 64 --tta`:

| Block | 14 | 15 |
|---|---|---|
| `r22689_c8230` avenue | 81.6% | 73.8% |
| `r28160_c16065` junction | 89.7% | 89.1% |
| `r18804_c10012` residential | 93.1% | 93.1% |
| `r21686_c31355` woodland | 93.5% | 93.4% |

- Avenue: most of the bicycle parking is "other" (patchy, not grid-aligned), which accounts for most of the drop.
  The lower median is partly sidewalk, which no earlier model managed.
- Junction: the level crossing is still predicted as road (rail 41%). The rail corridor south-east of it is fine.

## 16_wide_inference: 512 px models run on 768 px windows

`visualize.py --size 768`: the 14 and 15 models with 768 px windows (61 m) at prediction time instead of 512,
same weights. Files are `<checkpoint>_768_*`.

| Block | 14 | 14 at 768 | 15 | 15 at 768 |
|---|---|---|---|---|
| `r22689_c8230` avenue | 81.6% | 82.0% | 73.8% | 76.3% |
| `r28160_c16065` junction | 89.7% | 89.7% | 89.1% | 88.9% |
| `r18804_c10012` residential | 93.1% | 93.1% | 93.1% | 93.0% |
| `r21686_c31355` woodland | 93.5% | 93.3% | 93.4% | 93.1% |

- Avenue: the extra context removes some of the "other" on the bicycle parking; the lower median looks the same
  as at 512 in both models (road in 14, partly sidewalk in 15).
