# Can detail experiments

These are the rounded observations from the original sweeps. I used the public Can masks to choose the final recipe. The [integrated rerun](can_local_contrast.json) records the full precision results for that recipe. The quarter-size export in the historical log measured 60.9% under regular light. A later check of the current implementation at the same nominal weight and scale gave 61.46% under regular light and 59.65% across all six lights.

Every variant measured against the same public-split harness (`summarise_by_light`, AU-PRO @ 5% pixel FPR, mean over six lights). Reference configuration unless noted: tiled ResNet-18 layer2+3, 256-pixel native tiles with 25% overlap, uniform 4,096-vector bank, Gaussian smoothing sigma 2.0, good regular-light training images only, seeds as marked.

## Reference floors

| Map | AU-PRO | tiny-PRO | Note |
| --- | ---: | ---: | --- |
| Trivial brightness `L` | 0.0 | 0.0 | no information about defects |
| Trivial darkness `255 − L` | 25.8 | 39.6 | defects are dark |
| Trivial Sobel magnitude | 26.2 | 46.5 | the strongest no-learning floor |
| Trivial local contrast (blur−rad x) | 20.7 | 45.9 | |
| Trivial black top-hat (31 px) | 19.8 | 36.2 | |

## Deep baseline and detail branch

| Variant | AU-PRO | tiny | worst light |
| --- | ---: | ---: | ---: |
| Full frame | 0.0 | 0.0 | — |
| Full frame + basic lighting aug | 15.0 | — | — |
| Tiled, seed 42 (control) | 39.2 | 60.4 | 32.6 |
| Tiled, seed 7 | 36.9 | 52.3 | 33.4 |
| Tiled, seed 1234 | 35.2 | 51.3 | 29.2 |
| Control + dark branch, w 0.75 | 48.7 | 66.7 | 46.1 |
| Control + dark branch, w 0.5 | 49.1 | 67.4 | 47.5 |
| Control + dark branch, w 1.0 | 49.3 | 73.1 | 48.2 |
| Control + dark branch, w 1.25 | 44.7 | 69.1 | — |
| Control + bright branch, w 0.75 | 22.1 | 44.8 | 17.5 |
| Control + both-signed branch, w 0.75 | 42.7 | 65.0 | 40.4 |
| Combiner trained on synthetic defects | 31.4 | 39.9 | — |
| Control, artifact-masked | 39.2 | 60.4 | — |
| Branch fusion, weight plateau | 0.5–1.0 | — | — |

## Ensemble (3 seeds: 42, 7, 1234; mean of maps)

| Variant | AU-PRO | tiny | worst light |
| --- | ---: | ---: | ---: |
| Ensemble only (no branch) | 40.6 | 54.5 | 35.7 |
| Ensemble + dark branch φ | 49.8 | 66.9 | 48.3 |
| Ensemble + bright branch (sheet_metal) | 47.3 | 44.7 | 43.7 |
| Ensemble + bright branch (can) | 22.1 | 44.8 | — |
| Fabric, ensemble only | 38.1 | 51.1 | 34.4 |
| Sheet_metal, ensemble only | 44.8 | 41.5 | 42.5 |
| Wallplugs, ensemble only | 40.2 | — | — |
| Walnuts, ensemble only | 59.1 | — | — |

### Full selected configurations (local public split, after calibration)

| Category | Single-seed archive | Ensemble archive | Δ |
| --- | ---: | ---: | ---: |
| can (dark branch) | 49.2% | 49.6% | +0.4 |
| fabric | 36.1% | 38.1% | +2.0 |
| fruit_jelly | 36.9% | 38.8% | +1.9 |
| rice (dark branch) | 24.2% | 24.5% | +0.3 |
| sheet_metal (bright branch) | 45.6% | 47.3% | +1.6 |
| vial | 46.3% | 46.9% | +0.6 |
| wallplugs | 36.5% | 40.2% | +3.7 |
| walnuts | 58.0% | 59.1% | +1.1 |

Both archives pass MVTec's official checker. Ensemble files: 8,180 (4,090 `.tiff` continuous maps downscaled to 1/4 native resolution + 4,090 binary `.png` masks calibrated on good validation pixels at the 99.5th percentile).

## Reference bank size

| Bank | AU-PRO |
| --- | ---: |
| 4,096 | 39.9 |
| 32,768 | 35.1 |
| 131,072 | 37.3 |

## Resolution / feature level

| Configuration | AU-PRO |
| --- | ---: |
| layer2 + layer3 (this project) | 39.9 |
| layer1 + layer2 + layer3 | 37.7 |
| layer2 + layer3 at 2× input scale | 23.4 |

## Backbone and architecture, not just weights

| Configuration | AU-PRO | Note |
| --- | ---: | --- |
| ResNet-18 (this project) | 39.2–39.9 | reference |
| DINOv3 ViT-S/16 (frozen, blocks 8/10/11, kNN bank) | ~0.0 | flat map |
| DINOv3 ViT-S/16, blocks 2, 5, 8 | ~0.0 | no defect separation |
| DINOv3 ViT-S/16, block 5 only | inverted | defects lower than background |
| DINOv3 ViT-S/16, block 11 (raw) | ~0.0 | |
| Released SuperADD ViT-H+ bank, this split | 5.0 | published recipe, not this failure |
| Reduced SuperADD ViT-S/16 @ 0.625 scale | 6.4 | |
| Reduced SuperADD ViT-S/16, native scale | 11.1 | |

## Position / background subtraction variants

| Configuration | AU-PRO |
| ---: | ---: |
| φ Spatial residuals (per-location statistics) | 16.1 |
| φ Black top-hat, larger kernel (61 px) | 12.8 |
| φ Local contrast, window 201, branch | 49.1 (plateau) |
| φ Local contrast, window 101 | 38.6 |
| Per-cell PaDiM-style diagonal Gaussian | 0.0 (numerical failure) |
| Retinex log-luminance input transform | 37.0 |
| Golden-template per-pixel | 6.9–10.5 |
| Per-position config (fit on good tiles) | 0.0 |

## Cross-category transfer

| Category | Control | With branch | Best |
| --- | ---: | ---: | --- |
| can, dark branch | 39.2 | 48.7 (49.8 ens) | dark |
| sheet_metal, bright branch | 42.9 | 47.3 (ens) | bright |
| fabric, any branch | 36.1 | 38.1 (ens only) | no branch helps |

## Negative results we keep on record

Every measured approach that failed to beat its parent configuration: larger banks, higher resolution, position-conditioned Gaussian models, learned combiners trained on synthetic defects, Retinex-style normalization alone, and the frozen DINOv3 bank. The tables record the controls alongside the selected variants.

## Best measured configuration: local z-score + morphology (seed 42)

Branch: `local_zscore` radius 25, eps 0.5, dark polarity; post-processing: 5×5
maximum filter, per-image `(map − median) / (p99.5 − median)`; export at 1/2
native resolution. The code default now implements this recipe.

| Variant | AU-PRO | regular | worst light | tiny |
| --- | ---: | ---: | ---: | ---: |
| Control (deep only) | 39.2 | 39.9 | 32.6 | 60.4 |
| Frozen submission (box r=201, w=0.75) | 48.7 | 49.3 | 46.1 | 67.8 |
| Box r=501, w=0.5 | 51.2 | 51.8 | 50.0 | 68.3 |
| z-score r=25, w=0.5 | 53.0 | 54.3 | 50.3 | 72.5 |
| z-score r=25, w=0.5 + max5 | 60.1 | 61.4 | 57.5 | 81.8 |
| + per-image norm, export 1/4 | 59.1 | 60.9 | 55.8 | 79.3 |
| + per-image norm, export 1/3 | 61.1 | 62.5 | 58.2 | 80.8 |
| **+ per-image norm, export 1/2** | **62.9** | **64.1** | **60.4** | **81.7** |
| **same, w=0.75** | **63.9** | **64.4** | **61.9** | **81.6** |

Branch swap before morphology, all seeds (box r=201 w=0.75 → z r=51 eps=1.0 w=0.5):
seed 42 48.7 → 51.6; seed 7 48.1 → 51.8; seed 1234 42.8 → 45.0. The z-score
branch alone scores 28.6 mean AU-PRO, above Sobel (26.2) and darkness (25.8);
the old box branch alone was 18.9, below both.

The complete public-split polarity/weight sweep with the new post-processing
(dilation 5, per-image norm, 1/2 export) is:

| Polarity | w=0.5 | w=0.75 | w=1.0 |
| --- | ---: | ---: | ---: |
| dark | 62.9 | **63.9** | 62.8 |
| bright | 30.6 | 23.7 | 17.6 |
| both | 59.5 | 61.8 | 60.4 |
| control | 39.5 | — | — |

The existing selection rule picks `dark_w0.75`. The `both` arm is much stronger
after dilation than it was before it (42.7), but still 2.1 points behind dark;
the 14/16-dark prior justifies the one-sided choice.

Paired region test (96 region-light pairs): z25 vs control 73/96 improved,
p=3.1e-7; vs box r=201 59/96, p=0.032; vs box r=501 51/96, p=0.61.

Post-processing variants that did not help: grey closing 5 (55.2), extra
Gaussian σ=2 after fusion without dilation (49.5), dilation 4 (60.7),
dilation 6 (61.8), deep-map high-pass (27.6–37.7), unsharp masking (50.5–50.9).
Multiscale tile 128 + 256 (pre-morphology) gave 51.6 mean and +3.0 tiny over
tile 256, but was not combined with this recipe.
