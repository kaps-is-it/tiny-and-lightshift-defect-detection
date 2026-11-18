# Part decisions from tiled heatmaps

Tiled PatchCore localized some tiny defects on `can`, but its largest heatmap value rarely separated good and defective parts. We tested averaging the highest pixel scores and suppressing responses that also occur on good images or across broad areas.

All four scores used the same tiled detector, training images, and public evaluation scenes. The spatial method fitted a per-location median and robust spread on half the good regular-light validation images. Every method set its own 90th-percentile rejection threshold on the other half. The `can` split used 23 images for each role. We used the earlier `can` results to motivate the experiment, then fixed the methods before running them on `sheet_metal`.

| Score | How it works |
| --- | --- |
| Raw maximum | Use the highest pixel score in the original heatmap. |
| Raw top tail | Average the highest 0.01% of pixel scores. |
| Spatial top tail | Compare each pixel with its regular-light normal score distribution, then average the strongest 0.01% of positive residuals. |
| Local contrast top tail | Subtract the mean score in a 65-pixel square around each pixel, then average the strongest 0.01% of positive residuals. |

## Can

| Score | Part AUROC ↑ | Map AU-PRO at 5% ↑ | Missed defective captures ↓ | Good captures falsely rejected ↓ |
| --- | ---: | ---: | ---: | ---: |
| Raw maximum | 38.0% | 39.2% | 88 / 90 | 21 / 72 |
| Raw top tail | 40.8% | 39.2% | 83 / 90 | 15 / 72 |
| Spatial top tail | 60.7% | 16.1% | 69 / 90 | 14 / 72 |
| Local contrast top tail | 47.2% | 20.6% | 64 / 90 | 21 / 72 |

Spatial scoring improved the ranking by 22.8 points and reduced both errors at this operating point. Local contrast caught five more defective captures than spatial scoring and falsely rejected seven more good captures. It still missed 64 of 90 defective captures. The raw heatmap gave the strongest localization. A system can use the raw map to show an inspector where to look and use a separate score to decide which parts need attention.

AUROC and AU-PRO are averaged across the six lights. Error counts pool repeated captures of 15 defective and 12 good scenes, so the 162 decisions come from 27 independent scenes. The raw maximum row uses the same threshold subset as the other rows. The original README baseline used all 46 normal validation images to set its threshold.

## Sheet metal

| Score | Part AUROC ↑ | Map AU-PRO at 5% ↑ | Missed defective captures ↓ | Good captures falsely rejected ↓ |
| --- | ---: | ---: | ---: | ---: |
| Raw maximum | 87.2% | 42.9% | 44 / 90 | 0 / 24 |
| Raw top tail | 84.2% | 42.9% | 33 / 90 | 0 / 24 |
| Spatial top tail | 81.4% | 21.9% | 17 / 90 | 17 / 24 |
| Local contrast top tail | 86.9% | 31.4% | 32 / 90 | 0 / 24 |

Local contrast caught 12 more defective captures than the raw maximum at this threshold, with zero observed false rejects for both. Spatial scoring rejected every good capture under three of the shifted lights. The public split has only four good scenes, so a single extra false reject moves a per-light rate by 25 points. The regular-light validation split used nine images to fit the spatial reference and ten to set thresholds.

Local contrast is the next candidate for the part decision. It reduced missed captures without adding false rejects relative to the raw maximum in both experiments. Keep the original tiled map for localization. Test the decision rule on more independent scenes before using it as the default.

## Reproduce

```sh
python compare_scoring.py can
python compare_scoring.py sheet_metal
```

The two commands write per-image scores, thresholds, split filenames, and per-light metrics under `artifacts/score-comparison/`. These are local public-split comparisons.
