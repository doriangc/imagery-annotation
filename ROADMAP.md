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
- **Second multi-town run** (`data/unet_multi_v2.pt`): as the first, plus `--lighting`, on the farmland and
  paved-island labels. 2 h 23 min (25% slower: new pod with Python 3.14, or the lighting augmentation).

  | | Houten | Utrecht | Leidsche Rijn | Amsterdam | Nijmegen | Veere | **Amersfoort (test)** |
  |---|---|---|---|---|---|---|---|
  | mIoU | 0.681 | 0.522 | 0.494 | 0.549 | 0.430 | 0.602 | **0.544** |
  | with TTA | 0.651 | 0.525 | 0.509 | 0.550 | 0.450 | 0.622 | **0.545** |

  | | other | road | bike lane | sidewalk | parking | landscaping | traffic island | speed bump | rail | farmland | stairs |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | val | 0.81 | 0.72 | 0.58 | 0.50 | 0.51 | 0.59 | 0.38 | 0.37 | 0.83 | 0.92 | 0.00 |
  | test | 0.80 | 0.68 | 0.60 | 0.49 | 0.46 | 0.67 | 0.52 | 0.42 | 0.81 | - | 0.00 |

  - Test mIoU 0.479 → 0.544, but mostly from the labels: dropping agri (0.015) alone makes the first run 0.526
    over the same 10 classes. Landscaping +0.05, traffic island +0.04, speed bump +0.08; road, sidewalk,
    parking unchanged. Validation mIoU is the same as the first run (0.563).
  - TTA adds +0.015 on val but only +0.001 on Amersfoort.
  - Lighting, separated from the labels by scoring the first model on the new labels (both without TTA):

    | | val mIoU | Amersfoort mIoU | road | bike lane | parking | landscaping | speed bump | rail |
    |---|---|---|---|---|---|---|---|---|
    | first run (`unet_multi.pt`) | 0.568 | 0.521 | 0.67 | 0.57 | 0.45 | 0.65 | 0.34 | 0.75 |
    | + lighting, new labels (`unet_multi_v2.pt`) | 0.563 | **0.544** | 0.68 | 0.60 | 0.46 | 0.67 | 0.42 | 0.81 |

    So of the 0.479 → 0.544 gain on Amersfoort, about 0.04 is the relabelling and 0.02 the second run (lighting
    plus training on the new labels). The 0.02 is spread over 8 of 9 classes, so it's likely real; on the
    training towns there is no gain (−0.005, within noise). That fits lighting helping with an unseen photo
    flight more than with towns it has seen.
- **Context view** (`--context-scale 4`): each 384 px window also gets the same 384 px around the same centre at
  32 cm (123 m across), from `data/<area>_aerial_2026_ctx4.npy` (block means, built on first use). A second
  ImageNet ResNet-34 encodes it; its two deepest feature maps are resampled to the window's part and added to the
  U-Net's two deepest levels through zero-initialised 1x1 convolutions, so training starts as the plain U-Net.
  Aimed at the avenue medians (`r22689_c8230`), which only make sense with the whole street profile in view.
  Result (`data/unet_context.pt`, otherwise identical to `unet_multi_v2.pt`; 3 h 33 min on an A4000 vs 2 h 23 min,
  46M parameters vs 24M):

  | | val mIoU | Amersfoort mIoU | road | parking | landscaping | traffic island (val / test) |
  |---|---|---|---|---|---|---|
  | `unet_multi_v2.pt` | 0.563 | 0.544 | 0.68 | 0.46 | 0.67 | 0.38 / 0.52 |
  | + context view | 0.567 | 0.549 | 0.70 | 0.51 | 0.64 | 0.34 / 0.53 |

  - +0.004 / +0.005 mIoU: within noise. Parking +0.05 and road +0.02 on Amersfoort, landscaping −0.035.
  - The branch is used (the fusion weights grew well away from zero), but it does not fix the avenue: medians
    become sidewalk instead of road, and two reference blocks get worse (avenue 76.5% → 61.1%, woodland
    88.9% → 76.2%, with window-shaped patches).
  - Not adopted: 50% more training time for no clear gain. The code stays, off by default.
- **Avenue medians, label check** (`results/10_median_labels/`): training towns label real medians as traffic
  island too, so it isn't a convention difference. Amersfoort's tree-lined medians between carriageway and parking
  look like what other towns call sidewalk (trees in pits next to parking), and that design is rare in training.
  More variety (towns with boulevards) is the likely fix; the model and labels are not obviously at fault.
- **ConvNeXt encoders** (`overnight.sh`, `data/unet_convnext_tiny.pt`, `data/unet_convnext_small.pt`): the
  `unet_multi_v2.pt` setup with `--encoder tu-convnext_tiny` / `tu-convnext_small` (ImageNet-22k pretrained,
  fine-tuned on 1k), all else equal. No TTA in the table.

  | encoder | parameters | time (A4000) | val mIoU | Amersfoort mIoU | road | bike lane | parking | traffic island (val / test) |
  |---|---|---|---|---|---|---|---|---|
  | ResNet-34 (`unet_multi_v2.pt`) | 24M | 2 h 23 min | 0.563 | 0.544 | 0.68 | 0.60 | 0.46 | 0.38 / 0.52 |
  | ConvNeXt-Tiny | 32M | 3 h 32 min | 0.576 | 0.550 | 0.69 | 0.61 | 0.48 | 0.40 / 0.53 |
  | ConvNeXt-Small | 54M | 5 h 03 min | 0.582 | 0.546 | 0.68 | 0.57 | 0.48 | 0.40 / 0.49 |

  - Validation +0.013 / +0.019, possibly real but on towns the model has seen. On Amersfoort +0.006 / +0.002,
    within the noise between runs.
  - Reference blocks (`results/11_*`, `results/12_*`) are not better: the avenue drops to 60.8% / 70.0% (76.5%),
    with the medians almost entirely sidewalk, and both show a straight-edged rectangle of "other" over the
    bicycle parking, like the context view did.
  - Not adopted: 1.5x / 2.1x the training time for no gain on the unseen town. ResNet-34 stays the encoder, which
    also keeps runs short for adding towns.
- **Boulevard towns** (`areas.py`): `den_haag` (Koningskade, Zeestraat), `rotterdam` (Coolsingel, Weena,
  Westersingel), `eindhoven` (Kennedylaan, Fuutlaan), ~3 x 3 km each, for more wide streets with medians.
  `label_report.py` shows no new conventions for roads or fietsstraten, but Rotterdam maps paved medians as berm
  (so sidewalk) where Amersfoort and Houten use verkeerseiland.
- **Traffic island merged into sidewalk** (`rasterize.py`, class 7 now `unused_7`): paved islands look like
  sidewalk from above, towns label them differently (above), and the distinction can be recovered afterwards
  from position (paved strip between carriageways). Planted islands stay landscaping. For comparison with
  earlier models, `--evaluate ... --merge traffic_island=sidewalk` scores an old checkpoint the same way, on
  the old labels.
  Baseline, `unet_multi_v2.pt` on Amersfoort: mIoU 0.544 → **0.548** merged (10 classes instead of 11);
  sidewalk 0.486 → 0.495 with the islands counted in. The next run is compared against 0.548.
- **Boulevard run** (`data/unet_boulevards.pt`): v2 setup on nine towns with the merged labels, 50,000 steps,
  1 h 27 min on a B300 MIG slice (18 SMs; `nvidia-smi` shows no utilization for MIG slices).

  | Amersfoort | mIoU | road | bike lane | sidewalk | parking | landscaping | speed bump | rail |
  |---|---|---|---|---|---|---|---|---|
  | `unet_multi_v2.pt`, merged scoring | 0.548 | 0.68 | 0.60 | 0.50 | 0.46 | 0.67 | 0.42 | 0.81 |
  | `unet_boulevards.pt` | **0.559** | 0.70 | 0.60 | 0.50 | 0.44 | 0.67 | 0.47 | 0.85 |

  - +0.011 on the unseen town, from road, speed bump and rail; sidewalk unchanged. Levelled off (45k: 0.558).
  - Houten validation drops (≈0.66 → 0.59): a smaller share of training, and the most suburban town.
  - Reference blocks (`results/13_unet_boulevards/`): avenue 82.7% → 80.0%, other three up 0.4–4.6 points.
    The avenue's carriageways are cleaner, but one median is still road.
  - The straight-edged "other" blocks on the avenue's bicycle parking (also in 09, 11, 12) lie exactly on the
    window grid (rows 64/128/192 at stride 64, 128 at stride 128): stitching is fine, but the model's answer there
    depends on what each window happens to contain. Fed 768 px windows at prediction time (same weights), the
    blocks mostly disappear and the avenue goes 80.2% → 82.0%; the other three reference blocks are flat
    (88.4/88.4, 92.8/92.7, 92.8/91.9). So the model is short of context in places, and was never trained to use more.
- **Fietsstraat ignore fix** (`rasterize.py`): a road polygon was ignored when any OSM fietsstraat ran through it
  for 3 m, so roads a fietsstraat only crosses (junctions, bridges, its end) were ignored too, e.g. the main road
  of junction `r28160_c16065`. Now the overlap must also be at least 0.3 of the polygon's length (half its
  perimeter). Checked on photos: below 0.3 mostly crossings; 0.3–0.5 mostly short pieces of the route itself.
  Un-ignores 0–16,000 m² per town (Nijmegen most; Amersfoort 88 → 77 polygons). Not yet in any training run.
- **Level crossings as rail** (`rasterize.py`): BGT's `overweg` is the track corridor where rails cross a road,
  bike path or service path (checked on photos in Utrecht and Amersfoort: rails visible in every sample), not the
  road surface. As road it made roads seem to run along the track (junction `r28160_c16065`). Now rail. Small:
  74 polygons in Utrecht, 19 in Amersfoort, 0 in Houten, mostly under 60 m². Not yet in any training run.
- **512 px windows** (`data/unet_512.pt`, `--size 512`): the boulevard run with 512 px windows (41 m, 15 m of
  context on each side of the 10 m target) instead of 384 px (31 m, 10 m). Same labels, steps and towns.
  2 h 27 min on the B300 MIG slice (1.7× the 384 run).

  | Amersfoort | mIoU | road | bike lane | sidewalk | parking | landscaping | speed bump | rail |
  |---|---|---|---|---|---|---|---|---|
  | `unet_boulevards.pt` (384) | 0.559 | 0.70 | 0.60 | 0.50 | 0.44 | 0.67 | 0.47 | 0.85 |
  | `unet_512.pt` | **0.565** | 0.71 | 0.61 | 0.51 | 0.45 | 0.68 | 0.51 | 0.81 |

  - +0.006 on the unseen town and +0.006 on validation (0.583 → 0.589), small but across almost every class and
    every validation town except Amsterdam (equal). Rail drops, but rail is a few large patches in Amersfoort.
  - Reference blocks (`results/14_unet_512/`, scored on the new labels like `13/new_labels_*`): avenue 80.0% →
    81.6%, junction 88.9% → 89.7%, residential and woodland flat. The grid-aligned "other" block on the bicycle
    parking shrinks to the top-left corner; the lower median is still road. At 41 m the window still doesn't
    span the ~50 m avenue.
- **New labels run** (`data/unet_new_labels.pt`): the 512 run retrained on all ten towns re-rasterized with the
  fietsstraat fix and level crossings as rail. Baseline: `unet_512.pt` scored on the new labels, 0.589 val /
  0.566 Amersfoort (unchanged, as the fixes touch little area).

  | new labels | val mIoU | Amersfoort | road | bike lane | sidewalk | parking | speed bump | rail |
  |---|---|---|---|---|---|---|---|---|
  | `unet_512.pt` (old labels) | 0.589 | **0.566** | 0.71 | 0.61 | 0.51 | 0.45 | 0.51 | 0.81 |
  | `unet_new_labels.pt` | **0.590** | 0.563 | 0.71 | 0.60 | 0.50 | 0.44 | 0.50 | 0.82 |

  - A tie, within run-to-run noise. The fixes are kept because the labels are more correct, not for the score.
  - The junction's level crossing is still road (rail 41%, same as models that never saw it as rail): too few
    examples (~100 small polygons across ten towns), and the rails there are thin lines in asphalt.
  - Avenue 81.6% → 73.8%: the bicycle parking is now mostly "other", in irregular patches rather than window-grid
    blocks, and the lower median is partly sidewalk for the first time. So the avenue swings several points
    between runs that score the same on the town; read single-block changes with that in mind.
- **Wider windows at prediction time** (`visualize.py --size 768`, `results/16_wide_inference/`): both 512 models
  on 768 px windows (61 m, spanning the avenue), same weights. Avenue 81.6% → 82.0% and 73.8% → 76.3%, other
  blocks flat (±0.2). The gain is the bicycle parking: less "other". The lower median does not change in either
  model, so seeing the whole avenue doesn't fix it. More examples of this kind of median (paved strip with trees
  between a carriageway and a parking lane) look like the better lever than more context.
- **Review** (October 21): a written review of the next steps is kept as a doc, "bgt-fit model review". Its plan:
  measurement first, then dense supervision, then loss and learning-rate tuning, targeted sampling, infrared and
  AHN inputs, and cross-section aggregation along the road.
- **Measurement** (`center_patch.py`):
  - `--full-eval` scores every pixel of the validation strips and test areas (less a 256 px border), tiled with
    the model's target squares, after training or with `--evaluate`. The 1,600 evaluation windows per area cover
    about 2% of a town.
  - `--ema DECAY` evaluates and saves an exponential moving average of the weights; the raw weights are scored
    once more at the end for comparison.
  - `--seed` for repeat runs. Validation windows stay fixed whatever the seed.
  - `groningen` added to `areas.py` as a second test town, unlike Amersfoort. Its BGT is consistent with the
    others (`results/label_consistency.md`), apart from two fixes in `rasterize.py` that apply to every town:
    - `rijbaan autoweg` (expressway carriageway) is now road. It was unmapped, so it rasterized as "other":
      0.1% of road surface in Amsterdam, 0.9% in Nijmegen, 3.6% in Groningen.
    - Speed bump polygons over 1,500 m² are ignored: two in Groningen (5,757 and 1,811 m², whole road stretches),
      none elsewhere.
- **Dense supervision**: the loss used only the central 128 px of each window (6% of a 512 px window), a leftover
  from the label-inpainting setup. `--target 384` with `--size 512` supervises 9x more pixels per step with a
  64 px (5 m) context border. Evaluation windows are now scored on their central 128 px (`SCORE`) whatever the
  target, so earlier and dense runs stay comparable. `--evaluate --infer-target 128` predicts a dense model's
  central squares only, for more context per pixel.
- **Baseline on every pixel** (`unet_new_labels.pt`, `logs/baseline_full.log` on the pod):

  | | evaluation windows | every pixel |
  |---|---|---|
  | Amersfoort | 0.563 | **0.554** |
  | Groningen | 0.457 | **0.451** |
  | Houten val | 0.661 | 0.581 |
  | val pooled | 0.590 | 0.590 |

  - The windows were close for the test towns, but 0.08 off for Houten's validation strip. Per-area window
    scores can't be trusted at that level; compare every-pixel numbers from now on.
  - Groningen is 0.10 below Amersfoort, mostly bike lane (about 0.42 vs 0.61) and parking (about 0.32 vs 0.45).
  - Compare per town, not the pooled test mIoU: Groningen's few farmland patches (about 0.5 ha) score 0.000 on
    every pixel and cost the pooled mean about 0.05. Amersfoort has none.

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
3. **Pretrained encoder.** ImageNet ResNet-34 done; ConvNeXt-Tiny/Small tried, no better on Amersfoort. See
   the encoder section below for what's next.
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
