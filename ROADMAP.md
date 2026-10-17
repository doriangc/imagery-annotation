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

## Next steps

1. **Imagery as input**
   - Feed RGB crops instead of one-hot labels, with nothing hidden.
   - Larger target square: e.g. a 256 px input with a 128 px target.
   - Every scored pixel keeps at least the model's receptive field of context on all sides.
2. **90° rotations and flips** (all 8 combinations)
   - Rotate square crops before cutting the window.
   - Compare with and without, since rotation removes the consistent shadow direction.
3. **Pretrained encoder.** See the encoder section below.
4. **A second town as a test set** (BGT, labels and imagery), held out entirely. This measures generalisation
   beyond Houten's street design and its single photo flight, which the validation strip cannot.
5. **More towns for training.** Variety, not pixel count, is what limits the dataset.
6. **Rare classes.** Weight classes evenly for now. Later, re-weight the loss or oversample windows containing
   speed bumps, stairs and traffic islands.
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

1. A U-Net with an ImageNet-pretrained ResNet-34 or ConvNeXt-T encoder, as the baseline.
2. The same model trained from scratch, to measure what pretraining adds.
3. DINOv3, both the web and satellite variants, frozen with a light decoder.
   - On CPU: compute the features once and store them, then train only the decoder.
   - Fine-tuning the encoder needs a GPU.
4. Compare on the Houten validation strip and on the second town.

Expectation: pretraining mostly buys faster convergence and robustness in other towns. Expect a few mIoU points
over from-scratch, and a smaller gap between the ImageNet CNNs and DINOv3.

## Known limitations

- **Label noise we can't fix:**
  - BGT edges are only accurate to about 20–30 cm.
  - Buildings lean in the photos, so the roof edge doesn't line up with the BGT ground footprint.
  - Tree canopy covers paths.
  - BGT and the photos may date from different times.
- **Unfetched elevated layers:** where an elevated object we didn't fetch (e.g. a bridge) passes over a road,
  the labels show the road.
- **JPEG only:** the WMS serves JPEG, so the imagery is compressed one extra time.
- **No GPU:** everything runs on CPU so far.
