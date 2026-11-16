# Testing the part score on fabric and wall plugs

The can and sheet-metal experiments showed that reducing broad responses could improve part decisions even when it made the heatmaps less useful. Here, the detector keeps its raw tiled heatmap for localization. The local-contrast filter is used only to calculate a part score.

`fabric` and `wallplugs` are the transfer test. They give us a textured material and scenes with overlapping manufactured parts. Neither category was used to choose the scoring rule. Each category gets its own PatchCore memory bank from good regular-light training images.

Both scores use the same 256-pixel tiles, 4,096-feature bank, and raw heatmap for a given image. The baseline uses the largest value in that map. The candidate subtracts the local mean in a 65-pixel square, clips negative residuals to zero, and averages the highest 0.01% of the resulting values. Both thresholds are set at the 90th percentile of their respective scores on the same good regular-light validation images. The threshold stays fixed for all six public lighting conditions.

The fabric test has 11 good and 15 defective scenes. The wall-plug test has 10 good and 15 defective scenes. Each scene contributes six captures. False rejects and misses count captures. AUROC is averaged across lighting conditions, and its paired bootstrap interval resamples whole scenes within the good and defective groups. AU-PRO comes from the shared raw heatmap. The previews show that raw overlay with the local-contrast part decision in the caption.

## Reproduce

```sh
python run_experiment.py fabric --mode tiled --compare-part-scores
python run_experiment.py wallplugs --mode tiled --compare-part-scores
python summarize_hybrid.py fabric wallplugs
```

The per-category runs write paired scores, thresholds, metrics, and previews under `artifacts/<category>/tiled-paired/`. The summary writes `artifacts/hybrid_summary.json`.
