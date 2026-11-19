# What the dataset images preserve under changing light

A detector can miss a tiny scratch because the image carries too little local contrast, because preprocessing removes spatial detail, or because the part score ranks glare above the scratch. We tested how resizing changes the recorded defect signal, whether a simple image-quality monitor notices acquisition changes on good parts, and whether four verified-good references improve automated decisions under the changed light. The public MVTec AD 2 views let us compare the same scene under different exposure and lighting settings. [MVTec AD 2 paper](https://link.springer.com/article/10.1007/s11263-026-02743-0)

## How much detail survives resizing

For every defective capture, we measured the absolute difference between mean luminance inside its annotated defect mask and in the adjacent three-pixel ring. We measured it at the original size and after resizing the long edge to 256 pixels. Fractional mask coverage at the lower resolution lets us measure how much area remains even when a defect spans less than one output pixel. Both measures describe the decoded image and depend on the capture's annotation, texture, and color. A lower contrast proxy or the absence of a majority-covered pixel alone cannot establish that a defect is impossible to detect.

The median luminance difference fell after resizing in all **47 category and lighting groups** across the eight categories. Under regular light, Can's median marked area was 129 native pixels and projected to an effective 1.59 output pixels. Nine of 15 defective Can scenes had zero output pixels with at least 50% mask coverage. The median luminance difference on a 0–1 scale went from **0.092 to 0.027**. For comparison, Sheet Metal fell from 0.297 to 0.082 and Vial from 0.275 to 0.216. These results describe how this resize changes the signal in the existing images. [Lighting audit data](lighting_audit.json)

## Acquisition alerts on known-good parts

The monitor measures median luminance, the 5th-to-95th-percentile luminance range, and the fraction of decoded pixels near black or white. For each category, one half of the good regular-light validation images sets the feature reference. The disjoint half sets 90th-percentile alert thresholds for individual scenes and for the median score of four good scenes. Fitting uses no defective images, masks, or shifted captures. In the public replay, each four-scene combination uses the same parts under regular and changed light. Combinations overlap, so their counts describe sensitivity to the choice of reference scenes. They do not represent independent batches.

| Category and light | Good scenes alerted | Four-scene replay batches alerted |
| --- | ---: | ---: |
| Can, regular | 1 / 12 (8.3%) | 0 / 495 (0.0%) |
| Can, Shift 2 | 1 / 12 (8.3%) | 9 / 495 (1.8%) |
| Can, Shift 3 | 12 / 12 (100%) | 495 / 495 (100%) |
| Rice, regular | 0 / 7 (0.0%) | 0 / 35 (0.0%) |
| Rice, each of five changed lights | 7 / 7 (100%) | 35 / 35 (100%) |
| Fabric, regular | 4 / 11 (36.4%) | 203 / 330 (61.5%) |
| Fruit Jelly, each available light | 0–1 / 5 (0–20%) | 0 / 5 (0.0%) |

Batch alerts increased relative to regular light in 27 of the 39 changed conditions. Four categories produced regular-light batch alerts. Fabric had a high false-alert burden and Fruit Jelly showed no batch response. The monitor flags changes in acquisition statistics. Its alert counts cannot tell whether defect recognition improved or deteriorated. [Lighting audit data](lighting_audit.json) · [Audit script](../experiments/baselines/lighting_study.py)

## What four reference parts do to Can decisions

We took every four-part combination among the 12 verified-good Can scenes, for 495 support sets. Every view of a support scene stays out of both the fixed and recalibrated evaluations. Each set leaves eight good and 15 defective scenes under each of five changed lights. For the native-scale fusion score, we compared the fixed regular-light threshold with the median paired score offset and a robust affine fit with its positive scale bounded to 0.5–2. Both calibrations use paired regular and shifted captures of the four support scenes only. Their monotone transforms move the operating point while preserving within-light score ordering.

| Policy on changed-light Can | Mean good captures automatically rejected, out of 40 | Mean defective captures automatically passed, out of 75 | Mean captures referred, out of 115 |
| --- | ---: | ---: | ---: |
| Fixed fusion threshold | 10.00 | 54.00 | 0.00 |
| Median-offset calibration | 8.27 | 61.26 | 0.00 |
| Bounded affine calibration | 11.88 | 56.88 | 0.00 |
| Review the batch on a drift alert, fixed threshold otherwise | 1.98 | 10.80 | 92.42 |

Neither recalibration policy improved both mean errors for any of the five stored detector scores. For the fusion score, median offset added misses in 431 of 495 support sets. The review policy referred **80.4% of changed-light captures** and none of the regular-light captures in this replay. Its lower automatic counts reflect deferred decisions. Referred captures have no measured final outcome. These results do not justify an automatic threshold update on an alert. The offline comparison records how much referral would defer and which automatic errors remain. [Support replay data](support_adaptation.json) · [Replay script](../experiments/baselines/support_adaptation.py)

## Reproduce

```sh
python -m experiments.baselines.lighting_study can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
python -m experiments.baselines.support_adaptation artifacts/multiscale/native/can --lighting-audit artifacts/lighting_study/can
```

The first command saves per-capture CSVs and a summary for each category under `artifacts/lighting_study/`. The second replays the saved native-scale Can predictions and saves its report alongside them.
