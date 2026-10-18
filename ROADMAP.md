# bgt-fit roadmap

## Goal

Predict road cross-sections from 8 cm aerial imagery: for each pixel across a road, which of road, bike lane,
sidewalk, parking, landscaping (and the other classes below) it is. BGT supplies the training labels;
at inference time only imagery is available, no BGT.

## Done

- **`bgt_fetch.py`**: wegdeel, ondersteunendwegdeel and begroeidterreindeel for Houten from the PDOK OGC API,
  pickled to `data/houten_bgt.pkl`.
- **`rasterize.py`**: burns the BGT polygons into a `uint8` class raster in RD New (EPSG:28992), in 30 m tiles.
  - The default is 8 cm, on the aerial photos' pixel grid: the origin is snapped to even metres, which are
    multiples of 0.08 m.
  - Output: `data/houten_classes_8cm.npy` (51088 × 52900) plus a `.json` with the grid and class table.
  - Where objects overlap, the higher `relatieve_hoogteligging` wins, since that is what is visible from above.
- **Classes:** 0 other, 1 road, 2 bike_lane, 3 sidewalk, 4 parking, 5 landscaping, 6 driveway,
  7 traffic_island, 8 speed_bump, 9 rail, 10 verge_paved, 11 agri_natural, 12 stairs.
  - BGT covers the ground without gaps, so "other" is buildings, water, yards, bare ground, etc. It is a real
    class that the model is trained on.
- **`aerial_imagery_fetch.py`**: PDOK `2026_orthoHR` aerial photos on exactly the label raster's grid.
  - Output: `data/houten_aerial_2026.npy` (RGB, 8.1 GB).
  - The WMS snaps requests to its own 8 cm pixels instead of resampling. Alignment checked: tile seams and the
    label overlays match.
- **`center_patch.py`**: tests the training setup with labels as input.
  - Input: a 160 × 160 window with the central 32 × 32 hidden; the target is the hidden square.
  - Model: a dilated CNN that sees about ±63 px; validation is the southern 15% of the raster.
  - Baseline: copy the nearest visible pixel to the left of the hidden square.
  - Result after 2000 steps (18 min on CPU): 98.2% accuracy / 0.70 mIoU, vs copy-left 95.4% / 0.66.
    - Large gains on road, sidewalk, bike lane and parking.
    - Speed bump scores 0 (copy-left 0.97): with even weighting the model never predicts this rare class.
      See next step 6.
- **Imagery runs** (`center_patch.py`, 256 px input, 128 px target, random rotations/flips; RunPod RTX A4500).
  All scores are on the southern validation strip.

  | Model | Steps × batch | Time | Accuracy | mIoU | Road | Bike lane | Sidewalk | Parking |
  |---|---|---|---|---|---|---|---|---|
  | Small dilated CNN, from scratch | 2,000 × 16 | 1 min | 78.6% | 0.20 | 0.48 | 0.00 | 0.03 | 0.07 |
  | U-Net ResNet-34, from scratch | 10,000 × 32 | 13 min | 85.4% | 0.34 | 0.62 | 0.22 | 0.18 | 0.37 |
  | U-Net ResNet-34, ImageNet | 10,000 × 32 | 13 min | 88.8% | 0.46 | 0.72 | 0.44 | 0.41 | 0.59 |
  | U-Net ResNet-34, ImageNet | 50,000 × 32 | 54 min | 88.6% | 0.51 | 0.78 | 0.76 | 0.48 | 0.62 |

  - At equal training, ImageNet pretraining adds +0.11 mIoU, and roughly doubles bike lane, sidewalk and parking.
    That's more than expected; part of it is faster convergence, so the gap may narrow with longer training.
  - Both U-Nets were still improving at 10,000 steps.
  - At 50,000 steps (channels-last + compile, ~64 s per 1,000 steps):
    - mIoU levelled off at 0.511 from step 40,000 on, while the training loss kept dropping. The model is starting
      to fit the training area rather than generalise, so more steps on Houten alone won't help much.
    - Accuracy stayed at 88.6%; the gains are in smaller classes: bike lane 0.44 → 0.76, traffic island 0.79,
      rail 0.87, and driveway is non-zero for the first time (0.05).
    - Speed bump, paved verge and stairs are still 0.
  - The GPU runs at ~100% utilisation and 2.8 GB of 20 GB memory, so data loading keeps up.
- **`visualize.py`**: predicts map blocks by overlap-tiling and renders photo | BGT | prediction | errors.
  On validation blocks, the ImageNet U-Net:
  - gets roads and parking rows right, and finds bike lanes
  - misses narrow footpaths between back gardens
  - draws blobby boundaries
  - never predicts driveways
  - confuses paved plazas with sidewalk, other and parking

## Multiple towns (in progress)

- **Areas** (`areas.py`): Houten plus 3 × 3 km of Utrecht (centre/east), Leidsche Rijn, Amersfoort, Amsterdam
  (De Pijp/Rivierenbuurt), Nijmegen and Veere (town + countryside). Amersfoort is the held-out test town.
- **Why:** the Houten model (0.512 mIoU on Houten's val strip) scores 0.191 mIoU on Utrecht.
- **Label consistency** (`results/label_consistency.md`), and what we changed because of it:
  - Fietsstraten: Houten labels 23% of their length as bike path and Amsterdam 31%; other towns use carriageway.
    Road polygons under OSM fietsstraten are now ignored (255) in training and scoring.
  - Driveways: Utrecht/Leidsche Rijn barely map `inrit`, so driveways are merged into sidewalk.
  - Verges: Amsterdam maps mostly unpaved verges (tree pits); other towns mostly traffic islands or planted
    verges. Paved verges are now sidewalk, all others landscaping.
  - Level crossings (`overweg`) are road. Class ids 6 and 10 are kept as empty slots.
  - Class 11 is now **farmland** only (arable land, agricultural grassland, orchards, tree nurseries). Woods,
    shrubs, other grassland, reeds and marsh are landscaping. Before, "agri_natural" was farmland in Houten and
    Veere but mostly urban woodland in Amersfoort, which the model called landscaping (0.02 IoU on Amersfoort).
  - Traffic islands are now paved islands only; planted ones are landscaping (Houten's islands were 58% planted,
    most cities' under 10%), the same rule as verges. Prompted by Amersfoort's wide paved medians
    (`results/06_unet_multi/*r22689_c8230*`), which the model splits between sidewalk, road and island.
- **Lighting augmentation** (`--lighting`): random brightness, contrast, gamma, saturation, a slight colour cast,
  and half the time a soft-edged synthetic shadow, on training windows only. For shadowed and differently lit
  streets; the 2026 photos are leaf-off with long tree and building shadows.
  - Still to watch: speed bumps cover 0.05% of Houten vs 0.7–0.8% of Utrecht and Amersfoort.
- **Imagery:** Utrecht, Leidsche Rijn, Amersfoort and Amsterdam come from 5 cm photos that the WMS downsamples
  smoothly to our 8 cm grid; the others are native 8 cm.
- **Class weighting, Houten only** (old 13-class labels, 50,000 steps, 256 px, 100 val batches):
  `--class-weight-power 0.25` gives 0.548 mIoU vs 0.512 unweighted, at the same accuracy (87.3% vs 87.6%).
  Speed bump 0 → 0.27, stairs 0 → 0.16, driveway 0.03 → 0.13, sidewalk and parking +0.03; bike lane
  0.75 → 0.67, rail 0.94 → 0.88. Still improving at the end, unlike the unweighted run.
- **Larger windows:** a Houten-only 384 px run died at step 15,000 (DataLoader workers out of `/dev/shm`
  while another run shared the pod; prefetch is now 2 batches per worker). Not retested in isolation:
  384 px was adopted for the multi-town run directly. It costs ~2.3× time per step (137 s per 1,000 steps).
- **First multi-town run** (`data/unet_multi.pt`): U-Net ResNet-34 ImageNet, 384 px input, 128 px target,
  class weight power 0.25, 50,000 × 32, new labels, 1 h 54 min.

  | | Houten | Utrecht | Leidsche Rijn | Amsterdam | Nijmegen | Veere | **Amersfoort (test)** |
  |---|---|---|---|---|---|---|---|
  | mIoU | 0.611 | 0.482 | 0.502 | 0.525 | 0.452 | 0.595 | **0.479** |

  Val (all towns) 85.4% / 0.563 mIoU, test 81.9% / 0.479. The Houten-only model scored 0.191 on Utrecht.

  | | other | road | bike lane | sidewalk | parking | landscaping | traffic island | speed bump | rail | agri | stairs |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | val | 0.81 | 0.71 | 0.60 | 0.51 | 0.52 | 0.56 | 0.34 | 0.41 | 0.83 | 0.89 | 0.00 |
  | test | 0.80 | 0.68 | 0.59 | 0.48 | 0.46 | 0.62 | 0.48 | 0.34 | 0.81 | 0.02 | 0.00 |

  - The gap between training towns and Amersfoort is small (0.56 vs 0.48), so the model mostly learns what
    the classes look like rather than one town's labelling.
  - mIoU stalled until the OneCycle LR peak (step 15,000) and rose as it decayed; still rising slightly at 50,000.
  - Agri/natural scores 0.02 on Amersfoort: its "agri_natural" is mostly urban woodland, the training towns'
    is farmland. Fixed in the labels by making class 11 farmland only (see above).
  - Stairs are never predicted. Traffic island dropped from 0.85 (Houten only) to 0.34 on val.
  - Not comparable with the Houten-only numbers: different classes, masked fietsstraten, more towns.
- **Prediction-time averaging** (`visualize.py --stride/--tta`), same `unet_multi.pt` on 10 random Amersfoort
  street blocks (768 px, farmland labels; CPU time per block):

  | | plain | overlapping windows | rotations/flips | both |
  |---|---|---|---|---|
  | mIoU | 0.571 | 0.575 | 0.587 | 0.588 |
  | accuracy | 79.1% | 79.4% | 80.5% | 80.4% |
  | time | 3 s | 15 s | 23 s | 112 s |

  - Rotation/flip averaging gives +0.016 mIoU, mostly parking, landscaping and traffic island (+0.03–0.05).
  - Overlapping windows remove the visible seams between 128 px squares but add little to the score.
  - Most blobs remain: they are where the model is unsure (e.g. paved medians), not stitching artefacts.

## Next steps

Immediate, with the ImageNet U-Net:

- ~~**Train longer**~~ Done: 50,000 steps gives 0.51 mIoU and levels off from 40,000.
- ~~**Time `--channels-last` / `--compile`** on the pod~~ Done: per 500 steps, 51 s plain, 38 s with
  channels-last, 33 s with both (plus ~55 s of compiling once). Use both for long runs. The plain baseline varied
  between runs (77–102 s per 1,000 steps), so treat the exact percentages as rough.
- ~~**Larger windows**~~ 384 px input with the same 128 px target, used from the multi-town run on.

Broader steps:

1. ~~**Imagery as input**~~ Done: 256 px input, 128 px target.
2. ~~**90° rotations and flips**~~ Done (all 8, on by default). A run with `--no-augment` for comparison is
   still open; rotation removes the consistent shadow direction.
3. **Pretrained encoder.** ImageNet ResNet-34 done; see the encoder section below for what's next.
4. ~~**A second town as a test set**~~ Done: Amersfoort, held out entirely.
5. ~~**More towns for training.**~~ Done: six training towns, 0.191 → 0.479 mIoU on an unseen town.
6. **Rare classes.** Slight weighting (power 0.25) is on. Stairs are still never predicted; oversampling
   windows containing them is the next option.
7. **Cross-section extraction.** Sample windows aligned with road centrelines so the centre column runs
   across the road.
   - This needs arbitrary-angle rotation, which is deliberately postponed.
   - When it comes: render the labels from the vectors in the rotated frame, so only the imagery is
     interpolated.

## Pretrained encoders

| Encoder | Notes |
|---|---|
| ImageNet ResNet-34/50, ConvNeXt-T/S (`timm`, `segmentation_models_pytorch`) | **First choice.** At 8 cm, ordinary photo features transfer well (cars, markings, kerbs). Multi-scale features suit a U-Net. Feasible on CPU at small sizes. |
| DINOv2 (Meta, Apache 2.0) | Strong frozen features, but one feature per 14 px patch (≈1.1 m). Needs a decoder that restores fine detail. |
| DINOv3 (Meta, 2025) | Has a satellite-trained variant, but at much coarser resolution than 8 cm. Check the license for commercial use. |
| SAM / SAM 2 (Meta) | Finds object outlines, not classes. Useful for labelling tools, not for the main model. |
| Remote-sensing models (SatlasPretrain, Clay, Scale-MAE, SSL4EO, Prithvi) | Mostly 0.3–30 m. Resolution mismatch, little expected over ImageNet. |

Plan:

1. ~~A U-Net with an ImageNet-pretrained ResNet-34 or ConvNeXt-T encoder, as the baseline.~~
   Done (ResNet-34): 0.46 mIoU at 10,000 steps.
2. ~~The same model trained from scratch, to measure what pretraining adds.~~ Done: 0.34 mIoU, so pretraining
   adds +0.11.
3. DINOv3, both the web and satellite variants, frozen with a light decoder.
   - On CPU: compute the features once and store them, then train only the decoder.
   - Fine-tuning the encoder needs a GPU.
4. Compare on the Houten validation strip and on the second town.

Expectation was a few mIoU points over from-scratch. The measured gap at 10,000 steps is +0.11, larger than
expected. The gap between the ImageNet CNNs and DINOv3 is still open.

## Known limitations

- **Label noise we can't fix:**
  - BGT edges are only accurate to about 20–30 cm.
  - Buildings lean in the photos, so the roof edge doesn't line up with the BGT ground footprint.
  - Tree canopy covers paths.
  - BGT and the photos may date from different times.
- **Unfetched elevated layers:** where an elevated object we didn't fetch (e.g. a bridge) passes over a road,
  the labels show the road.
- **JPEG only:** the WMS serves JPEG, so the imagery is compressed one extra time.
- **Hardware:** training runs on a rented RunPod RTX A4500. The local machine (CPU-only) is used for code
  changes, short smoke tests and visualisation.
