# Experiment log

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

Every measured approach that failed to beat its parent configuration: larger banks, higher resolution, position-conditioned Gaussian models, learned combiners trained on synthetic defects, Retinex-style normalization alone, and the frozen DINOv3 bank. Nothing here was hidden; each is reproducible from this repository.
