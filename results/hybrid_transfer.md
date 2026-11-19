# Testing the part score on fabric and wall plugs

The can and sheet-metal experiments showed that reducing broad responses could improve part decisions even when it made the heatmaps less useful. Here, the detector keeps its raw tiled heatmap for localization. The local-contrast filter is used only to calculate a part score.

`fabric` and `wallplugs` are the transfer test. They give us a textured material and scenes with overlapping manufactured parts. Neither category was used to choose the scoring rule. Each category gets its own PatchCore memory bank from good regular-light training images.

Both scores use the same 256-pixel tiles, 4,096-feature bank, and raw heatmap for a given image. The baseline uses the largest value in that map. The candidate subtracts the local mean in a 65-pixel square, clips negative residuals to zero, and averages the highest 0.01% of the resulting values. Both thresholds are set at the 90th percentile of their respective scores on the same good regular-light validation images. The threshold stays fixed for all six public lighting conditions.

The fabric test has 11 good and 15 defective scenes. The wall-plug test has 10 good and 15 defective scenes. Each scene contributes six captures. False rejects and misses count captures. AUROC is averaged across lighting conditions. The 95% intervals use 10,000 paired bootstrap draws of whole scenes within the good and defective groups, holding the fitted thresholds fixed. AU-PRO comes from the shared raw heatmap. The previews show that raw overlay with the local-contrast part decision in the caption.

## Results

| Category | Score | Part AUROC | Defective captures missed | Good captures falsely rejected |
| --- | --- | ---: | ---: | ---: |
| Fabric | Raw maximum | 74.6% | 57 / 90 | 12 / 66 |
| Fabric | Local contrast | 72.5% | 76 / 90 | 8 / 66 |
| Wall plugs | Raw maximum | 63.0% | 33 / 90 | 30 / 60 |
| Wall plugs | Local contrast | 60.1% | 61 / 90 | 11 / 60 |

At the validation-calibrated thresholds, local contrast missed more defective captures in both categories and falsely rejected fewer good captures. Under changed lighting alone, wall-plug AUROC rounded to 60.7% for both scores, while missed defective captures rose from 22 to 48 out of 75. A similar ranking score can still yield very different inspection decisions after a lighting change.

The paired AUROC differences were −2.1 percentage points for fabric (95% interval −17.5 to +12.4) and −2.9 for wall plugs (−20.2 to +13.0). The intervals are wide enough to include gains and losses. Mean raw AU-PRO at 5% pixel false-positive rate was 36.1% for fabric and 36.5% for wall plugs. The extra local scoring step took a median of about 53 ms and 48 ms per image respectively.

### Tiny defects

A scene enters this check if every 8-connected region in its native regular-light mask covers at most 0.01% of the image. Its six captures stay together in the same group.

| Category | Tiny-only scenes | Raw maximum misses | Local contrast misses |
| --- | ---: | ---: | ---: |
| Fabric | 7 | 29 / 42 | 32 / 42 |
| Wall plugs | 2 | 3 / 12 | 3 / 12 |

The lower miss counts previously observed on can and sheet metal did not carry over at this operating point. Use raw maximum as the default part score and keep local contrast available for controlled comparisons.

## Reproduce

```sh
python -m local_contrast.run_experiment fabric --mode tiled --compare-part-scores
python -m local_contrast.run_experiment wallplugs --mode tiled --compare-part-scores
python -m experiments.baselines.summarize_hybrid fabric wallplugs --dataset data/mvtec_ad_2
```

The per-category runs write paired scores, thresholds, metrics, and previews under `artifacts/<category>/tiled-paired/`. The summary writes `artifacts/hybrid_summary.json`.
