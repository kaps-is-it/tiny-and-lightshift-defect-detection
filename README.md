# Tiny Defects and Changing Light Conditions

**Factories can struggle to detect tiny defects when inspection conditions change.** Picture a production line checking thousands of parts. The vision system performs well in testing. Then a light gets replaced, a camera shifts slightly, or a more reflective material comes through. A good part can suddenly look damaged, and a real scratch can be harder to see.

I first got interested in industrial computer vision while looking through my previous employer’s portfolio. Reading about how factories handle those changes led me to MVTec’s inspection datasets. AD 2 stood out because it lets you follow the same part through several lighting conditions and see where a detector looks each time. [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

Could an inspector tell when a change in conditions was making its predictions less reliable and adapt without needing thousands of newly labeled defective parts? That question led to a PyTorch study of lighting changes using 1,084 public photos of 184 scenes across all eight AD 2 categories.

## My best result on Can

The best Can heatmap reached **67.53% localization AU-PRO** on public MVTec AD 2 by combining six aligned photos of each part. With one photo at a time, the same model reached **66.62% across the five changed lights** and **66.71% across all six**. AU-PRO measures how well the heatmap covers individual defects. The calculation here goes up to a 5% pixel false-positive rate.

The model looks at overlapping tiles at the photo’s original resolution. ResNet-18 features compare the surface with patches sampled from good cans, and a pixel-level score helps pick up small marks that are darker than their surroundings. I average the finished heatmaps from three independently sampled banks of good patches. With six photos, a second pixel-by-pixel average combines the six single-photo maps. A scratch that shows up in the same spot across several views keeps contributing to the result. [How I built and tested the ensemble](#12-one-can-under-six-lights) · [Per-light results](results/can_ensemble_multiview.json)

![Six crops of the same Can defect, their single-view heatmaps, and the six-view mean beside the annotation](results/visuals/can_ensemble_views.png)

*Each column shows the same 100 × 100 pixels of Can 003 under a different light, with a shared heatmap scale. On the right, the cyan outline marks the 131 annotated pixels and the panel below shows the six-view heatmap.*

## How my model compared

Since MVTec publishes the dataset and builds industrial inspection software, I wanted to see how my detector stacked up against established models on MVTec’s own benchmark images.

Every model below takes one photo per prediction. The project’s evaluator scored the maps on the same 162 public Can captures, including 135 under changed lighting. CAT and SuperADD use their released Can configurations. The Anomalib runs use 558 × 256 inputs, and their maps for good captures enter scoring at that smaller size.

| Single-image model on public Can · AU-PRO up to 5% pixel FPR | Five changed lights | All six lights |
| --- | ---: | ---: |
| **My three-bank ensemble** | **66.62%** | **66.71%** |
| Released CAT checkpoint | 33.87% | 34.86% |
| Anomalib PatchCore | 6.83% | 7.30% |
| Released SuperADD reference bank | 4.91% | 4.95% |
| Anomalib PaDiM | 2.14% | 2.20% |
| Anomalib STFPM | 2.08% | 2.09% |

The ensemble’s single-image heatmap scored **20.23 percentage points higher than my original tiled baseline** on the five changed lights. PatchCore had the strongest localization of the three Anomalib runs. [Ensemble results](results/can_ensemble_multiview.json) · [CAT and SuperADD settings](#7-comparison-with-released-cat-and-superadd) · [Anomalib experiment and scoring details](#13-testing-anomalib-on-can)

**Sheet Metal** gave me a different clue. On many missed parts, the heatmap already pointed near the flaw. Scoring those suspicious regions raised detections from **30 to 50 out of 75 defective captures** under changed lighting, with **three false rejects out of 20 good captures** in both runs. [How I traced the misses](#9-going-back-to-the-parts-i-missed)

## How I got there

If you’re curious how the model took shape, the full study is below. It follows the experiments from the first baseline to the missed images that gave me ideas for what to try next.

**Jump to** [The data](#2-the-data-and-how-i-used-it) · [How I measured progress](#5-how-i-measured-progress) · [Experiments](#6-the-experiments-in-order) · [Leaderboard context](#8-where-the-website-benchmark-fits) · [Local dark contrast](#11-the-can-scratch-that-kept-me-working) · [Three banks and six views](#12-one-can-under-six-lights) · [Anomalib comparison](#13-testing-anomalib-on-can) · [Reproduce the study](#reproduce-the-study)

### 1. The problem I investigated

#### What existing systems struggle with

Commercial machine vision and open-source anomaly detection are already well established. MVTec builds inspection software, and Anomalib provides model implementations that gave me a starting point for the research. The gap I wanted to understand was how reliably a detector could find small flaws after the inspection conditions changed. [MVTec machine vision](https://www.mvtec.com/) · [Anomalib](https://github.com/open-edge-platform/anomalib)

MVTec AD 2 was designed to make that kind of test possible. It includes tiny defects in large images, reflective surfaces, and repeated photos of the same scene under different lights. In the authors’ original 2026 journal comparison, the best evaluated method reached **58.7% mean localization AU-PRO** across the benchmark categories, with the curve integrated up to a 30% pixel false-positive rate. The project’s main tables use a 5% limit. [MVTec AD 2 journal paper, Table 1](https://link.springer.com/article/10.1007/s11263-026-02743-0)

An *IEEE Access* study published on September 21, 2026, found that **30.9% to 46.1%** of calibrated pass/fail decisions flipped across five tested configurations in paired comparisons of private images under regular and mixed lighting. That measures how consistently the detectors made decisions under a change of light. Determining which decisions were wrong would require the private defect labels. [IEEE Access lighting study](https://ieeexplore.ieee.org/abstract/document/11701006)

#### What the study set out to test

The goal was to see what would make inspection more stable when lighting changed. The plan brought together anomaly detection with self-supervised features, domain adaptation using a few known-good examples, threshold calibration, and selective human review. MVTec’s existing photos made it possible to test those ideas under real lighting changes on a laptop GPU.

A normal-image anomaly detector learns what good products look like and assigns high scores to regions that differ from its reference. That makes it possible to fit the detector when damaged examples are scarce. A harmless reflection can also differ enough from the reference to score highly.

The first step was to measure how much defect detail survived resizing, then check the heatmaps on missed parts against their annotations. The later experiments tested whether a monitor could flag a lighting change, whether a few good examples could help recalibrate the inspector, and which cases might benefit from review. The evaluation tracked three outputs separately. The heatmap should point to the damage, the image score should rank defective parts above good ones, and the threshold should make reasonable pass/fail decisions. The experiments showed that each can improve or deteriorate on its own.

### 2. The data and how I used it

The two MVTec datasets served different purposes. The original MVTec AD gave me a straightforward Bottle category for checking the pipeline. MVTec AD 2 supplied high-resolution images, small defects, and repeated views of the same scenes under real lighting changes. The dataset paper describes scenarios including reflective surfaces, transparent objects, overlapping parts, and tiny anomalies. [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [MVTec AD 2 paper](https://link.springer.com/article/10.1007/s11263-026-02743-0)

| Data | Role in the study |
| --- | --- |
| Original MVTec AD, Bottle | Verify loading, normal-reference fitting, thresholds, masks, and end-to-end evaluation. |
| AD 2 training | Fit each category's normal patch references using good regular-light images. |
| AD 2 validation | Set normal score scales and fixed thresholds using good regular-light images. |
| AD 2 public test | Measure localization and decisions against the available labels and masks. |
| AD 2 private images | Check local prediction generation and export format. Their labels are held by MVTec. |

The AD 2 inventory contains 2,528 good training images, 302 good validation images, 1,084 public test captures, and 2,045 images in each of the two private splits. The public evaluation spans **184 distinct scenes**, including 120 defective and 64 good scenes. Each scene has between four and seven lighting views.

| Category | Good scenes | Defective scenes | Lighting conditions | Public captures |
| --- | ---: | ---: | ---: | ---: |
| Can | 12 | 15 | 6 | 162 |
| Fabric | 11 | 15 | 6 | 156 |
| Fruit Jelly | 5 | 15 | 4 | 80 |
| Rice | 7 | 15 | 6 | 132 |
| Sheet Metal | 4 | 15 | 6 | 114 |
| Vial | 5 | 15 | 7 | 140 |
| Wall Plugs | 10 | 15 | 6 | 150 |
| Walnuts | 10 | 15 | 6 | 150 |

The manifest groups captures by category, good/defective folder, and scene number. The folder matters because the image counters restart for good and defective examples. All views of a reference scene are excluded from the corresponding adaptation evaluation. Each capture uses its own defect mask because visible annotation can change with lighting. The data loader checks for missing files and mask alignment. [Scene manifest builder](local_contrast/build_scene_manifest.py)

![One Can part under six dataset-provided lighting conditions with actual model responses](results/visuals/lighting_comparison.png)

*The same defect remains in the crop across six real dataset captures. Each overlay uses that capture's annotation and the same fitted native model. The score-to-threshold ratio is the image-level decision statistic. This selected scene is rejected under all six conditions.*

### 3. What the research changed in my approach

The research covered normal-reference methods, dense visual features, small-defect sampling, photometric processing, synthetic supervision, and adaptation. Published implementations and the MVTec leaderboard helped narrow down which methods could be compared directly.

| Research direction | Idea that carried over | Experiment |
| --- | --- | --- |
| PatchCore and patch memory banks | A pretrained representation of good patches can support both anomaly detection and localization. Representative reference coverage matters. | Uniform normal banks, projected farthest-first selection, raw maxima, and alternative part scores. |
| Dense self-supervised features | Strong general visual representations are promising, but feature stride and resolution still determine which details reach the scoring stage. | DINOv3 ConvNeXt-Tiny and the released transformer-based CAT and SuperADD models. |
| Fine and contextual features | Early features preserve local texture and edges. Deeper features provide a wider context. | Separate stride-four and stride-eight maps, then validation-scaled fusion. |
| Retinex and photometric processing | Smooth illumination correction might reduce exposure variation, while changing contrast can also alter defect evidence. | Bounded gamma, mild Retinex, and simulated lighting in normal training images. |
| Student–teacher methods | A student trained on normal appearance can expose unusual regions through prediction error. | A compact student predicting standardized frozen ResNet features. |
| Synthetic-defect learning | Good images can supply backgrounds for artificial scratches and spots. Transfer to real defects needs measurement. | A small segmentation branch trained on simulated defects, evaluated on real MVTec defects. |
| Few-reference adaptation | Verified-good shifted images can estimate changes in scores or features. | Paired score offsets, bounded affine calibration, feature correction, and a lighting-statistics monitor. |

PatchCore, DINOv3, EfficientAD, and DRAEM supplied useful methodological references. CAT provided a released prototype-reconstruction model. SuperADD provided a released normal-reference configuration with overlapping windows and a DINOv3 backbone. The papers gave me hypotheses to test on the same images. [PatchCore](https://arxiv.org/abs/2106.08265) · [DINOv3](https://arxiv.org/abs/2508.10104) · [EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) · [DRAEM](https://arxiv.org/abs/2108.07610) · [CAT](https://github.com/VisualScienceLab-KHU/VAND2026_CAT) · [SuperADD](https://arxiv.org/abs/2605.14808)

The review also covered reconstruction, diffusion, super-resolution, and vision-language approaches. I focused the implemented comparisons on methods that could produce spatially aligned maps with an interpretable compute budget. The student–teacher and synthetic branches are compact implementations of those research directions. Their measured results describe those configurations.

#### Measuring the loss before changing the detector

Can images are 2,232 × 1,024 pixels. In the 15 defective regular-light scenes, total marked defect area ranges from 25 to 450 native pixels. The median is 129. In the resizing audit, its marked area projected to an effective 1.59 output pixels when the long edge was reduced to 256.

The audit compared defect-to-neighborhood luminance contrast before and after resizing. The median contrast proxy fell in all **47 category-light groups**. For regular-light Can, it went from **0.092 to 0.027** on a 0–1 scale. Nine of 15 defective Can captures had no resized output pixel with at least 50% defect-mask coverage. These are measurements of the available image signal. Detector accuracy still needed a separate test. [Audit results](results/lighting_audit.json) · [Audit implementation](experiments/baselines/lighting_study.py)

This made resolution preservation my first implementation priority.

![Matched full-frame and native-tile model responses at the same source-image crop](results/visuals/resolution_comparison.png)

*The full-frame model sees the image after fitting it inside a 256-pixel square. The tiled model keeps native local detail. Both use the same 96 good source images and reference capacities. This comparison shows their restored responses over the same original-image crop.*

### 4. How I built the native-resolution inspector

The inspector uses a frozen ImageNet-pretrained ResNet-18 and fits a separate reference for each product category.

1. **Keep local image detail.** Split the image into native 256 × 256 windows with at least 25% overlap. The windows cover the full image, including the edges.
2. **Extract two representations.** The fine branch uses unpooled layer-one features at stride four. The context branch combines locally pooled layer-two and interpolated layer-three features on a stride-eight grid.
3. **Build compact normal references.** Retain 1,024 fine features and 4,096 context features through seeded uniform sampling. A patch’s anomaly score is its Euclidean distance to the nearest normal reference.
4. **Restore the source-image coordinates.** Resize each patch grid back to its crop, stitch overlapping maps with tapered weights, and apply a two-pixel Gaussian smoothing radius in native coordinates.
5. **Balance the two branches.** Divide each map by the median image maximum for that branch on good validation images, then average the scaled maps.
6. **Make a part decision.** The maximum fused score is compared with the 90th percentile of good validation image scores. The same threshold is used under each test light.

For context map (C), fine map (F), and normal-validation scales (s_C) and (s_F), the fused map is

```text
A(x, y) = 0.5 × [C(x, y) / s_C + F(x, y) / s_F]
part_score = maximum value in A
```

The anomaly score measures departure from the normal reference. The measured thresholds and errors show what those scores mean for inspection.

Can requires 60 overlapping tiles. At stride eight, those tiles generate 61,440 patch queries. A full FP32 distance matrix against 4,096 references would occupy roughly 960 MiB. Chunking the nearest-neighbor calculations keeps the distance buffer around 32 MiB. The smaller fine bank helps offset the fine grid’s fourfold query count. [Detector](experiments/baselines/multiscale_detector.py) · [Tiling, fusion, and evaluation runner](experiments/baselines/multiscale_experiment.py)

![Actual Can inspection with the source image, defect crop, fused heatmap, annotation, and thresholded prediction](results/visuals/inspection_hero.png)

*This is the first regular-light defective Can scene that native RGB fusion correctly rejected, `bad/003`. The cyan outline marks the 131 annotated pixels. The white region in the thresholded panel is the model’s prediction.*

![Context, fine-detail, and fused responses for the same annotated Can defect](results/visuals/feature_branches.png)

*Each branch is displayed relative to its own normal-validation threshold. Context and detail each pass this capture at their separate thresholds. Their fused score crosses its threshold. The image shows the actual branch outputs for this one case.*

### 5. How I measured progress

The main controlled comparison uses **96 evenly spaced good training images per category**. The backbone, input geometry, reference sizes, random seed, and evaluation captures stay fixed wherever the tested method permits. The lighting-augmentation variant adds a simulated view of each selected normal image. A separate fit using all 412 good Can images tests the effect of using more source images.

| Measure | What it tells me |
| --- | --- |
| AU-PRO at 5% pixel FPR | How well a map covers individual defect regions while limiting marked background. Higher is better. |
| Tiny-region overlap at 5% pixel FPR | Coverage of connected defect regions occupying at most 0.01% of the native image. |
| Image AUROC | How well image scores rank defective captures above good ones across thresholds. |
| Pixel F1 | How well a particular binary defect mask balances precision and recall. |
| Missed defects | Defective captures passed at the selected part threshold. |
| False rejects | Good captures rejected at that threshold. |
| Compute time | Preprocessing, inference, and stitching time for a source image under the recorded timing boundary. |

The AU-PRO implementation uses 4,096 histogram bins and integrates up to 5% pixel false positives. The localization averages weight each relevant lighting group equally. Eight-category summaries then weight categories equally. Part-error counts pool captures.

The bootstrap keeps repeated views together by scene. Fifteen defective Can scenes provide 90 captures, but uncertainty still depends on those 15 scenes. The denominators in the tables show how many captures went into each result.

The 90th-percentile rule supplies a source-data operating point. Its observed false-reject rate can change under a different lighting distribution.

### 6. The experiments, in order

#### First check on Bottle

The Bottle pipeline check used 167 good images to fit the original MVTec AD reference and 42 held-out good images for calibration. On 83 test images, it caught **all 63 defective examples** and rejected **3 of 20 good examples**. Image AUROC was 100% and local AU-PRO was about 71.5%. That gave me an end-to-end check before moving to the harder AD 2 cases.

#### The first Can resolution test

The early single-context-branch experiment used all 412 good Can training images and all six public lights.

| Early Can configuration | Mean AU-PRO | Misses | False rejects | Median compute |
| --- | ---: | ---: | ---: | ---: |
| Full frame | 0.0% | 73 / 90 | 13 / 72 | 30 ms |
| Full frame with brightness/contrast augmentation | 15.0% | 58 / 90 | 22 / 72 | 29 ms |
| Native tiles | 39.2% | 90 / 90 | 15 / 72 | 168 ms |

Tiling recovered localization, while the normal-derived part threshold passed every defective capture. That result motivated the separate work on part scoring and finer features. Later controlled runs use the 96-image budget and the two-branch architecture described above, so their numbers have their own tables.

#### Changing the score used for the part decision

The comparison covered raw maximum, top-tail averaging, spatial normalization, and local contrast. The spatial reference used one half of good validation images. The other half set thresholds for all four alternatives.

| Early Can score | Mean image AUROC | Map AU-PRO | Misses | False rejects |
| --- | ---: | ---: | ---: | ---: |
| Raw maximum | 38.0% | 39.2% | 88 / 90 | 21 / 72 |
| Top 0.01% of raw pixels | 40.8% | 39.2% | 83 / 90 | 15 / 72 |
| Spatially normalized top tail | 60.7% | 16.1% | 69 / 90 | 14 / 72 |
| Local-contrast top tail | 47.2% | 20.6% | 64 / 90 | 21 / 72 |

Local contrast subtracts the mean in a 65-pixel neighborhood and scores the strongest positive residuals. The Sheet Metal comparison reused the four scores, with nine normal validation images for the spatial reference and ten for threshold calibration.

| Early Sheet Metal score | Mean image AUROC | Map AU-PRO | Misses | False rejects |
| --- | ---: | ---: | ---: | ---: |
| Raw maximum | 87.2% | 42.9% | 44 / 90 | 0 / 24 |
| Top 0.01% of raw pixels | 84.2% | 42.9% | 33 / 90 | 0 / 24 |
| Spatially normalized top tail | 81.4% | 21.9% | 17 / 90 | 17 / 24 |
| Local-contrast top tail | 86.9% | 31.4% | 32 / 90 | 0 / 24 |

The Fabric and Wall Plugs runs kept the raw heatmap and used local contrast only for the part decision. Fabric misses rose from **57 to 76 of 90**, while false rejects fell from 12 to 8 of 66. Wall Plugs misses rose from **33 to 61 of 90**, while false rejects fell from 30 to 11 of 60. The result made transfer testing essential before adopting a score rule. [Part-score experiment](results/scoring_comparison.md) · [Transfer experiment](results/hybrid_transfer.md)

#### Ten controlled Can configurations

The following results cover the five changed lights. Each configuration is evaluated on 75 defective and 60 good captures. The map columns retain the context-only, fine-only, and fused outputs. Tiny overlap and decision counts use the fused output.

| Configuration | Context AU-PRO | Fine AU-PRO | Fusion AU-PRO | Fusion tiny overlap | Misses | False rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-frame ResNet | 2.6% | 9.5% | 5.8% | 3.7% | 66 / 75 | 7 / 60 |
| Native RGB ResNet | 35.9% | 35.5% | 46.4% | 73.1% | 54 / 75 | 15 / 60 |
| Bounded gamma | 35.0% | 33.8% | 43.6% | 72.8% | 55 / 75 | 20 / 60 |
| Mild Retinex | 36.0% | 34.5% | 46.9% | 74.2% | 66 / 75 | 16 / 60 |
| Simulated lighting in training | 34.3% | 31.1% | 40.3% | 63.8% | 63 / 75 | 10 / 60 |
| Projected farthest-first bank | 36.3% | 34.1% | 41.8% | 65.1% | 46 / 75 | 25 / 60 |
| DINOv3 ConvNeXt-Tiny | 6.2% | 21.4% | 16.3% | 16.5% | 65 / 75 | 5 / 60 |
| Sobel/Laplacian fine features | 35.9% | 16.0% | 33.0% | 53.9% | 65 / 75 | 13 / 60 |
| Synthetic-defect segmentation | 35.9% | 7.9% | 25.2% | 39.4% | 41 / 75 | 28 / 60 |
| Student–teacher fine features | 35.9% | 20.8% | 33.3% | 61.1% | 62 / 75 | 10 / 60 |

![Actual full-frame, native RGB, gamma, Retinex, and lighting-augmentation outputs on the same Can scene](results/visuals/can_trials_01.png)

![Actual coreset, DINOv3, derivative, synthetic-defect, and student–teacher outputs on the same Can scene](results/visuals/can_trials_02.png)

*All ten visual comparisons use `bad/000` under regular light. Each card contains a full-image fusion overlay, a crop from the native-coordinate map, and the actual score-to-threshold ratio. The source annotation contains 60 pixels. Cyan marks its boundary. The generator refits each configuration from its recorded good training paths and checks the regenerated scores against the saved predictions. The examples show individual behavior, while the table uses all changed-light captures.*

The result from Retinex was particularly instructive. Its map-level gain over raw RGB fusion was about half a percentage point, but it missed 12 more defective captures at its own validation-derived threshold. The synthetic branch caught more defective captures while rejecting many more good ones.

The farthest-first variant first samples a reservoir four times the final bank budget, projects the features into 16 dimensions, and selects diverse references. Its fused part score improved Can misses with additional false rejects. The context-only coreset score improved both counts on Can, reducing misses from 70 to 51 of 75 and false rejects from 9 to 6 of 60.

The DINOv3 ConvNeXt trial used stages 0, 1, and 2 at the same tile and reference budgets. The gradient trial substituted polarity-aware derivatives for the fine branch. The two trained branches used 180 optimization steps and up to 512 normal crops. Those budgets make the comparison reproducible and define the configurations behind the reported results.

Using all **412 good Can training images** with the same final bank sizes gave **43.8% fusion AU-PRO**, 56 misses, and 19 false rejects under changed lighting. The all-source fit scored lower than the 96-image fit at these fixed reference sizes. [All-source fit](results/can_full_fit.json) · [All configuration metrics](results/multiscale.json)

#### Transfer across all eight categories

![The same native fusion pipeline on the first defective regular-light example from every AD 2 category](results/visuals/category_examples.png)

*Each thumbnail shows the first defective regular-light capture in filename order for its category. The thumbnails retain the recorded input and fusion overlay. Every example is defective, so each displayed PASS is a missed defective capture. The category-level table below uses the complete changed-light evaluation.*

| Category | Context AU-PRO | Fusion AU-PRO | Context → fusion tiny overlap | Context → fusion misses |
| --- | ---: | ---: | ---: | ---: |
| Can | 35.9% | 46.4% | 50.5% → 73.1% | 70 → 54 / 75 |
| Fabric | 39.3% | 45.0% | 54.0% → 64.8% | 42 → 37 / 75 |
| Fruit Jelly | 37.0% | 31.8% | 44.3% → 44.7% | 39 → 40 / 45 |
| Rice | 20.9% | 21.6% | 56.3% → 96.4% | 59 → 62 / 75 |
| Sheet Metal | 45.9% | 43.5% | 44.6% → 42.2% | 41 → 45 / 75 |
| Vial | 45.0% | 38.9% | 94.6% → 96.7% | 80 → 87 / 90 |
| Wall Plugs | 36.7% | 34.9% | 65.3% → 67.9% | 33 → 75 / 75 |
| Walnuts | 57.1% | 57.4% | 100.0% → 100.0% | 24 → 48 / 75 |
| **Category mean** | **39.7%** | **39.9%** | **63.7% → 73.2%** | **388 → 448 / 585** |

Fusion had the highest AU-PRO of the three branch choices in four categories. Its strongest aggregate gain was tiny-region coverage. The fixed fusion thresholds missed more defective captures overall and falsely rejected fewer good captures, **43 versus 63 of 315**.

The coreset ran on Fabric and Wall Plugs as well. Fusion AU-PRO rose from **45.0% to 52.4%** on Fabric and from **34.9% to 39.6%** on Wall Plugs. Fabric fusion misses rose from 37 to 48 of 75, and Wall Plugs still passed all 75 defective shifted captures. A better map could still coincide with a poor part threshold.

![Localization across categories and the controlled Can configurations](results/multiscale.png)

#### Lighting alerts, adaptation, and review

The lighting monitor uses good regular-light validation images. It measures luminance median, luminance spread, and near-black/near-white fractions. Separate validation subsets set the reference and alert thresholds.

The monitor’s batch alert rate increased in 27 of 39 changed category-light conditions. Four categories also produced regular-light batch alerts. Rice alerted on every changed-light four-scene combination and none under regular light. Fruit Jelly produced no batch alerts in any light. The study tracks alert counts separately from defect recognition errors.

The first pilot used two or four reference scenes with the early full-frame Can baseline. On Shift 3, its held-out eight good and 15 defective scenes showed the trade-off clearly.

| Early full-frame Shift 3 policy | Misses | False rejects |
| --- | ---: | ---: |
| Fixed threshold | 7 / 15 | 3 / 8 |
| Two-reference offset | 7 / 15 | 4 / 8 |
| Four-reference offset | 11 / 15 | 0 / 8 |

The later fusion-score adaptation study replayed all **495 combinations of four good Can scenes**. All views of those four scenes were excluded from the scored sets, leaving eight good and 15 defective scenes per condition.

| Fusion policy over five changed lights | Mean misses, out of 75 | Mean false rejects, out of 40 |
| --- | ---: | ---: |
| Fixed threshold | 54.00 | 10.00 |
| Paired median offset | 61.26 | 8.27 |
| Bounded affine calibration | 56.88 | 11.88 |

Neither recalibration policy jointly improved average misses and false rejects for any of the five stored scores. The support combinations overlap, so their 495 outcomes describe sensitivity to reference choice.

A separate run applied half-strength, per-channel feature correction using four fixed good support scenes. On the remaining 115 captures, AU-PRO moved from **47.19% to 47.40%**, while misses rose from **54 to 61 of 75** and false rejects fell from **10 to 8 of 40**. Image AUROC moved from 48.83% to 48.33%.

Reviewing context/fine disagreement referred 47 of 135 shifted Can captures. It left 43 defective captures automatically passed. Random referral at the same workload would leave 35.2 in expectation. A broad drift-triggered policy referred 80.4% of the changed-light captures in the separate support replay. The replay measures both the review workload and the errors the automatic decisions still leave.

The last replay used MVTec’s paired views to simulate a second look. For one illustrative regular-plus-Shift-1 fusion comparison, rejecting if either view rejected reduced misses from 13 to 8 of 15, with 5 of 12 false rejects in both cases. It used an average of 1.74 dataset views per scene. The comparison uses the available images and sums their measured compute costs. [Adaptation results](results/support_adaptation.json) · [Feature correction](results/feature_adaptation.json) · [Paired-view and review analysis](results/paired_robustness.json)

![Two actual mistakes, a tiny defective Can passed and a good Can falsely rejected](results/visuals/failure_case.png)

*The left capture contains a 60-pixel annotated defect and receives a passing score. The right capture is labeled good but a bright response crosses the rejection threshold. These cases use the same native RGB fusion model as the successful example.*

### 7. Comparison with released CAT and SuperADD

The first comparison put native RGB fusion alongside the released Can models. All three ran on the same 162 public captures and used the same evaluator for their maps. CAT’s checkpoint includes its DINOv3 ViT-L/16 encoder. The run retains the authors’ four-by-four split, 448-pixel model inputs, cosine residual, uint8 tile quantization, and published binary cleanup.

The SuperADD run loads the authors’ complete Can bank with 100,000 embeddings at each of four ViT-H+/16 layers. Its configuration uses 0.625× image scaling, 640-pixel windows, and the saved mask threshold. The run uses the timm backbone conversion with the original rotary-period precision and chunks the nearest-neighbor queries. Code revisions and checkpoint hashes are saved with the results.

The table covers five changed lights.

| Method | AU-PRO | Tiny overlap | Pixel F1 | Misses | False rejects | Median compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **My native RGB fusion** | **46.39%** | **73.14%** | 2.89% | 54 / 75 | **15 / 60** | **215 ms** |
| Released CAT checkpoint | 33.87% | 42.00% | **4.21%** | **27 / 75** | 32 / 60 | 2,027 ms |
| Released SuperADD bank | 4.91% | 9.04% | 0.00% | 34 / 75 | 28 / 60 | 3,141 ms |

Native RGB fusion gave the strongest localization and tiny-region overlap in this comparison. CAT caught the most defective captures at its released operating point and had the highest pixel F1. SuperADD had the highest mean image AUROC, 60.44%, followed by CAT at 53.78% and native fusion at 45.11%.

The operating points differ. The native fusion threshold comes from normal validation maxima. CAT and SuperADD reject a part if their released binary mask contains any positive pixel. A separate CAT rule calibrated on normal validation images missed 24 defective captures and rejected 38 good ones. The numbers show the practical error trade-off alongside map quality.

The earlier reduced SuperADD runs reached 6.4% mean AU-PRO across all six lights at 0.625× scale and 11.1% at native scale. Tiny overlap fell from 17.2% to 9.1%. Those scale experiments used ViT-S/16 and a reduced reference budget. The full released bank was evaluated in a separate run. [Complete released-model comparison](results/released_comparators.md) · [CAT metrics](results/cat_release.json) · [SuperADD metrics and hashes](results/superadd_release.json)

### 8. Where the website benchmark fits

The **November 17, 2026** snapshot of the official leaderboard contains 1,039 entries. These are the highest posted AU-PRO values at the 5% pixel false-positive integration limit.

| Official quantity | Highest posted score | Entry |
| --- | ---: | --- |
| Can, regular private split | 94.68% | YL |
| Can, mixed-light private split | 80.97% | YL |
| Eight-category mean, regular private split | 88.72% | YL |
| Eight-category mean, mixed-light private split | 82.44% | Five tied DINO-DDB variants |

The feed also records **41.46%** mixed-private Can AU-PRO for CAT and **12.11%** for SuperADD. Those scores come from MVTec’s private images. The leaderboard provides context for the public Can comparisons, which use the same captures for all three methods. This project’s scores come from the public split. [Official leaderboard](https://benchmark.mvtec.com/) · [Dated snapshot and source fields](results/benchmark_context.json)

The YL entry provides a score with empty paper, author, and code fields in the retrieved feed. The tied DINO-DDB entries list LIN and provide empty paper and project fields. I selected CAT and SuperADD for reproduction because their released code and assets support a method-level comparison.

Metrics also change which method leads. SuperADD’s published VAND 4.0 results report **57.42% mean segmentation F1** on the regular private split and **54.35%** on the mixed private split. The website’s highest posted mixed-private mean segmentation F1 is **61.65%** from tied DINO-DDB entries. These values average eight categories. The local pixel-F1 values describe Can only. [SuperADD paper](https://arxiv.org/abs/2605.14808) · [VAND 4.0 organizer report](https://arxiv.org/abs/2610.04392) · [Leaderboard snapshot](results/benchmark_context.json)

The dataset paper’s original “below 60%” finding uses AU-PRO integrated up to **30%** pixel false positives. The project’s main tables and the leaderboard snapshot use **5%**. The integration limit is listed with each comparison because it affects the score. [Dataset paper and evaluation definitions](https://link.springer.com/article/10.1007/s11263-026-02743-0)

### 9. Going back to the parts I missed

After the first round, the baseline could produce a convincing heatmap on a can and still passed 54 of the 75 defective Can captures under changed lighting. I wanted to know where it had looked on those 54 images. If the scratch was among its strongest responses, a better part score had something to work with. If the strongest responses were all somewhere else, the heatmap itself needed work.

#### Was the defect among the five strongest responses?

I rebuilt the original inspector and reproduced its peak score on all 1,084 public captures. For each heatmap, the search drew a 33 × 33 box around the strongest pixel and repeated until it had five non-overlapping boxes. The annotations then showed whether any box touched a labeled pixel and whether any peak landed inside the defect.

![Two Can captures passed by the original inspector, with its actual five proposals and a magnified view of each defect](results/visuals/hotspot_shortlist.png)

*Each crop shows the first shifted baseline miss in filename order for its outcome. In the upper crop an orange box just reaches the cyan annotation. In the lower crop none of the five boxes reaches it. The boxes are proposed inspection areas. Their size is independent of the size of the defect.*

On Can, the five boxes covered a median **0.24% of each image**. They reached the defect in only **2 of the 54 missed captures**. Both were views of scene `bad/007`, and neither peak landed inside the annotation. Sheet Metal gave me a different answer. The boxes reached the defect in **26 of 45 misses**. There was more evidence there for a new decision rule to use.

#### What does a suspicious region on a good part look like?

The detector and heatmap stayed the same. For each category, **32 additional good training images** supplied reference hotspots. Each hotspot was described by its peak, how far it stood above the surrounding ring, its average excess across the box, and its position. The rule scores each proposed region against three similar good examples and uses the most unusual region as the part score. Both the old peak score and the new region score had their own 90th-percentile threshold fitted on good validation images.

| Changed lighting | Original peak rule | Normal-region rule | Difference |
| --- | --- | --- | --- |
| Sheet Metal, defective captures caught | 30 / 75 (40.0%) | 50 / 75 (66.7%) | **20 more caught**, up 26.7 percentage points |
| Sheet Metal, good captures rejected | 3 / 20 | 3 / 20 | Same count |
| All eight categories, defective captures caught | 137 / 585 | 151 / 585 | **14 more caught** |
| All eight categories, good captures rejected | 43 / 315 | 47 / 315 | **4 more false rejects** |

On Sheet Metal the region rule recovered 23 captures the old rule had passed and lost three it had caught. The 75 defective captures were five shifted views of 15 defective scenes. Resampling entire scenes gave a 95% interval of **+7 to +34** for the change in detections and **−3 to +3** for the change in false rejects. Of the 23 recovered captures, the region that triggered the decision touched the annotation in eight. That last count matters if someone is going to use the orange box to inspect a part.

![A recovered Sheet Metal capture, showing the source, the original crop and the model's leading inspection region](results/visuals/sheet_region_decision.png)

*This is the first shifted Sheet Metal capture in filename order where the old rule passed, the new rule rejected, and the leading box touched the annotation. Each ratio above the panels uses that rule’s own threshold.*

Across all eight categories the earlier local-contrast score caught 115 defective captures at 60 false rejects. The region score caught 151 at 47. Relative to the peak rule, Fabric lost nine detections and Walnuts lost eleven. On Wall Plugs the region rule caught eight more, and the leading box missed the annotated flaw in every one of those cases. These results helped me decide which modes to expose in the inspector. [Every category, threshold, and uncertainty estimate](results/hotspot_study.md#transfer-across-all-eight-categories)

#### Could I make the Can heatmap point somewhere better?

Three changes used the same 96 good source images. Shifting the tile grid by two pixels in each direction tested whether a scratch was falling between feature samples. Adding image coordinates to the patch features made distant pieces of normal texture less attractive matches. The lighting-aware variant estimated how much each context channel changed under simulated lighting on good crops and downweighted the unstable channels.

| Can, five changed lights | AU-PRO at 5% | Tiny overlap | Pixel F1 | Caught / 75 | False rejects / 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original native fusion | 46.39% | 73.14% | 2.89% | 21 | 15 |
| Average of two grid offsets | 46.47% | 72.03% | 2.18% | 16 | 10 |
| Maximum of two grid offsets | 46.94% | 72.37% | 1.87% | 15 | 6 |
| Position-aware matching | **47.74%** | **75.99%** | **8.02%** | 14 | 19 |
| Lighting-aware context | 47.29% | 73.94% | 3.23% | 21 | 16 |

**Position helped most with the mask.** Pixel F1 rose by **5.13 percentage points**, improving in each of the five changed lights. The close-up below shows the same can as the original inspector example. Under Shift 1, the old thresholded mask covered 32 of the 131 annotated pixels. With position-aware matching it covered 114. Both rules rejected this example. Across the full set, the position-aware rule caught seven fewer defective captures and rejected four more good ones. [Can measurements by lighting condition](results/can_sampling.json)

![Actual before-and-after Can heatmaps and thresholded masks, shown with the original crop and annotation](results/visuals/can_position_matching.png)

*Same source pixels, annotation and crop. The white model masks show which pixels exceeded each model’s normal-validation threshold. The 32 and 114 counts describe this one defect. The table includes false-positive pixels across all the evaluated images.*

The Fabric run reused the position settings with Fabric’s own normal references. AU-PRO fell from **45.00% to 37.20%** and tiny overlap fell from **64.81% to 54.46%**. Keeping track of where a patch came from helped on Can and hurt on Fabric. That transfer check is why I made position an option in the inspector. [Fabric transfer results](results/fabric_position_transfer.json)

#### Would a second detector spot something mine missed?

The two-model check combined proposals from the inspector and the released CAT model, scoring each box by how strongly both detectors responded inside it. Among the original 54 missed Can captures, the combined top five reached the defect in **9**, up from **2**. At the thresholds fitted on good images this rule caught **46 of 75** defective captures and falsely rejected **37 of 60** good ones. CAT’s released mask rule alone caught 48 at 32 false rejects. The two-model check took about **2.15 seconds per image**, compared with about 0.2 seconds for native fusion on Can. Getting a second opinion helped the shortlist, and CAT alone had the better part-decision trade-off in this run. [Model agreement results](results/can_cat_agreement.json)

### 10. What I kept after the miss analysis

I kept native RGB fusion as the default. It made the biggest difference to the original resizing problem and provided a consistent baseline across products. For **Sheet Metal decisions**, normal-region scoring gave the clearest improvement at the same observed false-reject count. For **Can localization**, position-aware matching gave the best mask at this stage. The right mode depended on whether the job was to draw the boundary of a flaw or to reject a part.

Here is an actual single-image run of the position-aware inspector. It fitted the normal Can references, read the Shift 1 image, and returned a score of **1.1174** against a threshold of **1.0087**. The preview and JSON decision came from the same command. The annotation was used separately to make the comparison figure above.

![Screenshot of an executed local notebook with inspection.py, the actual score and threshold, and its saved input and heatmap preview](results/visuals/inspection_session.png)

### 11. The Can scratch that kept me working

Position-aware matching took Can from 46.39% to 47.74% AU-PRO under changed lighting. The new map made it easier to point to a scratch and show which pixels the change recovered. The inspector still caught just 14 of 75 defective captures at its part threshold, so I went back to the images. Patch distance told me how different a patch looked. Looking at the source pixels raised a simpler question. Was the mark lighter or darker than the metal around it?

In the public Can images, 14 of the 16 annotated defect regions were darker than their surroundings under all six lights. That gave me something specific to try. The tiled detector could keep providing context, with a separate signal for thin dark marks at the original pixel resolution.

#### What I tried next

Scoring pixels by darkness alone reached **25.8%** mean AU-PRO across six lights. An edge detector reached **26.2%**. Both picked up real marks as well as ordinary printing. A learned combiner trained on synthetic defects reached **31.4%**. The tiled deep-feature control in this series reached **39.2%**.

| Can configuration | Mean AU-PRO across six lights | Change tested |
| --- | ---: | --- |
| Deep features + wide dark residual | 48.7% | Compared each pixel with its 201 × 201 neighborhood. |
| Wider dark residual | 51.2% | Tried a 501 × 501 window and a weight of 0.5. |
| Deep features + local dark z-score | 53.0% | Used a 25 × 25 window and divided the contrast by the local variation. |
| Same map + 5 × 5 maximum filter | 60.1% | Let strong edge responses reach neighboring pixels inside a flaw. |
| Per-image scaling and half-size score map | 62.93% | Rescaled each capture and kept more detail in the saved map than a quarter-size export. |
| Same settings, dark-detail weight of 0.75 | **63.94%** | Gave a little more weight to the pixel-level signal. |

The early rows are rounded measurements from the experiment log. The final recipe uses seed 42. Other runs tested both directions of contrast, extra smoothing and a larger reference bank. With the final postprocessing and weight of 0.75, bright-only reached **23.7%**, both directions reached **61.8%**, and dark-only reached **63.94%**. Earlier trials on Sheet Metal favored bright marks, which helped me decide to keep this recipe specific to Can. [Full experiment log](results/contrast_experiments.md)

#### How a Can image becomes a heatmap

1. **Learn what normal cans look like.** Cut 412 good training images into overlapping 256-pixel tiles. A frozen ResNet-18 extracts patch features from two layers. Keep a seeded sample of 4,096 normal feature vectors. At inspection time the distance to the nearest reference gives a coarse heatmap of unfamiliar patches.
2. **Measure the mark at its original size.** Compare each pixel’s luminance with the mean in a 25 × 25 window. Divide the difference by the local standard deviation plus 0.5 and keep positive values where the pixel is darker. A five-level dip stands out more on smooth metal than on busy lettering.
3. **Combine the two maps.** On 46 separate good validation images, measure the median and the distance from the median to the 99th percentile for each signal. Use these to put the signals on comparable scales. The fused score is the deep score plus **0.75 × the dark-detail score**.
4. **Fill narrow gaps and adjust for exposure.** A 5 × 5 maximum filter expands narrow responses. Subtract the map’s median and divide by its 99.5th percentile minus its median. This preserves the pixel ordering inside a capture while changing how scores from different captures compare at a shared threshold. Store the map at half width and height in float16 and resize it back to the annotation size for evaluation.
5. **Make the part decision separately.** Compare the maximum of the final map with the 90th percentile of the 46 good validation maxima. The earlier inspector used the same rule, with its own cutoff. Fitting the reference bank, map scales and part threshold takes only good images. The labeled public development scenes guided the choice of sign and weight and supplied the evaluation.

![Real Can image, ground-truth outline, deep patch response, dark detail and final fused score on the same 100-pixel crop](results/visuals/can_local_detail.png)

*This is Can 003 under Shift 1, the same 131-pixel defect shown in the position-aware comparison above. The cyan box shows the crop, and the cyan outline traces the annotation. The two branch panels share a display scale. The final panel shows the processed score in its own units. The dark spot inside the outline lights up alongside smaller marks above and below it. The part decision uses the highest score anywhere on the full image.*

#### How much better was it?

The final single-bank setting ran on the same **27 public Can scenes** under all six lights, giving **162 captures**. Each lighting condition gets equal weight in the averages below. AU-PRO measures how well the heatmap covers individual flaws at low pixel false-positive rates. Tiny overlap measures coverage of small annotated regions at the 5% pixel false-positive limit. Higher is better for both. The part-decision counts cover the five changed lights.

| Method | AU-PRO, all six lights | AU-PRO, five changed lights | Tiny overlap, changed lights | Defective caught / 75 | Good falsely rejected / 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier native fusion | 46.72% | 46.39% | 73.14% | **21** | **15** |
| Earlier position-aware matching | 48.17% | 47.74% | 75.99% | 14 | 19 |
| **Local dark contrast + deep features** | **63.94%** | **63.85%** | **81.61%** | 17 | 19 |

![Measured Can AU-PRO by light for native fusion, position-aware matching and the final local-contrast method](results/visuals/can_local_lights.png)

The gain over my previous best Can map was **16.11 percentage points** under changed lighting. Tiny-region overlap rose by **5.61 points** using the unrounded measurements. Even the hardest light for the new method, Shift 2, reached **61.87%**. This compares two complete systems. The earlier runs used 96 normal training images and this one used 412, so that difference is part of the comparison too.

The released CAT and SuperADD Can models scored **33.87%** and **4.91%** changed-light AU-PRO on these same public captures. CAT caught **48 of 75** defective captures at its released mask rule, with **32 of 60** good captures falsely rejected. The local-contrast model averaged **50.11% image AUROC** under changed lighting. Its heatmap got much better at showing where to look. The peak score was still a weak way to tell good cans from bad ones. [Public Can measurements](results/can_local_contrast.json) · [Per-capture decisions](results/can_local_decisions.json) · [Earlier Can runs](results/can_sampling.json) · [Released-model comparison](results/released_comparators.json)

![Full-frame input and heatmap from the Can inspector with its measured peak score, threshold and part decision](results/visuals/can_local_inspection.png)

*This is the full-image inspection preview for the same capture. The color bar shows the final anomaly score. The inspector can produce this view for an unlabeled image. The public annotation is shown separately in the zoomed-in figure.*

The **63.94%** is a public development result for **one category and one seed**. The repeated views contain 15 distinct defective scenes. Those public scenes helped me decide which version to keep. The private leaderboard scores come from a separate split. [Official benchmark context](#8-where-the-website-benchmark-fits)

### 12. One can under six lights

At 63.94% AU-PRO, the detector could pick up a small dark mark beside the printing on a can. Two questions were still bothering me. How much did the result depend on which normal patches happened to land in the reference bank? And what could the inspector learn from seeing the same can under a different light?

The first experiment still works with one photograph. The second needs all six aligned photographs of the same part, including the regular-light view.

| Public Can · AU-PRO up to 5% pixel FPR | Input per prediction | All six lights | Five changed lights | Worst light | Tiny overlap, all lights |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier local-contrast model, seed 42 | One capture | 63.94% | 63.85% | 61.87% | 81.57% |
| Three reference banks | One capture | 66.71% | **66.62%** | 65.26% | 81.64% |
| Three banks + scene mean | Six aligned captures | **67.53%** | **67.53%** | **67.53%** | **81.82%** |

#### First I changed the reference bank

The frozen ResNet sees the same 412 good training images each time. The seed determines which 4,096 patch features go into the bank, so a small scratch can get a different nearest neighbor in each fit. The three banks use seeds 42, 7 and 99, each calibrated on the same 46 good validation images. The runner keeps the 25 × 25 dark-contrast window, uses 1.0 in the local standard-deviation denominator and sets the detail weight to 0.75. It also applies an extra Gaussian blur (σ = 2 pixels) to the deep map before fusion. The three **finished** maps are averaged after the 5 × 5 maximum filter, per-image scaling and half-resolution round trip.

On their own, the three seeds reached **64.09%**, **64.99%** and **66.13%** all-light AU-PRO in the rerun. Averaging their maps reached **66.71%**. That gained **2.77 percentage points under changed lighting** over the published single-seed model. I chose the seeds after looking at the public results. Adding seed 3 to the three-bank setup gave 66.62% across all lights in the seed sweep. [Seed sweep](experiments/refinements/seed_sweep.py) · [Four-bank comparison](experiments/refinements/ensemble_selection.py)

#### Then I combined views of the same part

For each of the 27 public scenes, the six-view method averages the single-view ensemble maps at each pixel. A mark that stays in place can contribute six times. A reflection that moves or disappears contributes only in the views where it is visible. The dataset supplies the scene grouping and the captures are already aligned. The masks enter the scoring and the illustration after the mean map has been built.

The [six-view figure above](#my-best-result-on-can) shows Can 003, the same 131-pixel defect from the earlier examples. The shared color scale makes it easier to follow the strength of the response from one light to the next.

The extra views added **0.91 points** over the single-view ensemble on the changed-light measure. Across all six lights the total gain over the published single-seed model was **3.59 points**, and under the five changed lights it was **3.68 points**. The lowest per-light score rose by **5.66 points** over the single-seed run. All six per-light scores are identical because each scene gets the same output for every light and its six annotations are pixel-identical. There are still only **27 distinct scenes**, including **15 defective ones**. For a camera that supplies just one image, **66.62%** is the relevant changed-light result. [Full rerun and per-light measurements](results/can_ensemble_multiview.json) · [Published single-seed measurements](results/can_local_contrast.json)

**What happened to the part decision?** The single-view ensemble’s peak cutoff came from the good validation captures. It caught **6/75** defective changed-light captures and falsely rejected **8/60** good ones. Rejecting a scene if *any* of its five changed views crossed that cutoff caught **3/15 defective scenes** and falsely rejected **5/12 good scenes**. One bright response could decide the fate of an entire can. Applying the same cutoff (2.532) to the **averaged** map caught **0/15 defective scenes** and falsely rejected **1/12 good scenes**. The validation set supplies only regular-light images, so the six-view comparisons used the single-view cutoff. The 67.53% measures localization across a range of pixel thresholds. I would use the map to guide a reviewer.

#### What else I tried

The transfer and extra-data figures come from the original experiment sweeps. The three-bank comparison above was rerun for this release.

| Change on Can | Public AU-PRO, all lights / changed lights | What the result showed |
| --- | ---: | --- |
| Add the 46 validation images to training | 66.77% / 66.71% | A small change, and the calibration images now overlap training. The main experiment kept the original split. |
| Add the 72 good public test captures to training | 66.66% / 66.60% | A diagnostic that reuses evaluation inputs. Those good test captures were part of training for this run. |
| Augment normal training images with lighting changes | 65.28% / 65.24% | This augmentation reduced the score with the same bank capacity. |
| Mix local-contrast and native-fusion maps | 65.18% best across all lights | The three-bank average gave me a better Can map. |
| Subtract a per-cell normal average | 47.26% across all lights | Removing a location’s typical response also removed useful signal. |
| Adjust each map using its lowest-scoring pixels | 65.92% best across all lights | This simple within-image correction underperformed the unadjusted ensemble. |

Other runs tried a position-conditioned bank and two ways to score suspicious regions. The position-bank combination reached 0.00% in that configuration. One region rule caught 41/75 changed-light defects and falsely rejected 25/60 good captures. It traded more catches for more false alarms. These were lightweight experiments with the project’s features. A full prototype-reconstruction or subspace method would be a separate implementation. [Map and decision sweeps](experiments/refinements/region_decisions.py) · [Hybrid and position trials](experiments/refinements/hybrid_maps.py) · [Training-data trials](experiments/refinements/training_data.py)

The transfer results are **single-view** scores averaged over **all available lights for each category**. The earlier baseline uses the same averaging convention.

| Product | Earlier native fusion | New run | Configuration |
| --- | ---: | ---: | --- |
| Fabric | 45.56% | **54.24%** | Dark detail, seeds 42 and 7 |
| Sheet Metal | 44.84% | **52.60%** | Bright detail, weight 0.5, seeds 42 and 7 |
| Wall Plugs | 35.19% | **36.88%** | Dark detail, seeds 42 and 7 |
| Vial | **38.80%** | 34.35% | Dark detail, seeds 42 and 7 |
| Rice | 21.72% | **22.08%** | Single bank, seed 42 |

Fabric and Sheet Metal showed the clearest transfer. Vial is why I wouldn’t pick the Can recipe for every product. Bottle from the original MVTec AD served as a separate pipeline check. The two-bank result there was 48.65%. [Native-fusion baselines](results/multiscale.json) · [Category runs](experiments/refinements/parameter_sweep.py) · [Fabric and Sheet Metal](experiments/refinements/category_transfer.py) · [Bottle check](experiments/refinements/bottle_transfer.py)

The comparison at the top uses the single-photo ensemble so every model gets the same number of photos per prediction. The reproduction steps run both ensemble versions and save the maps for the six-view figure. [Single-photo comparison](#how-my-model-compared) · [Reproduce the study](#reproduce-the-study) · [Evaluation and scene averaging](local_contrast/multiview.py) · [Map construction](local_contrast/ensemble_maps.py)

### 13. Testing Anomalib on Can

HALCON is one of MVTec’s commercial machine-vision tools. For this experiment, I ran PatchCore, PaDiM, and STFPM from Anomalib, Intel’s open-source anomaly detection library. Its `MVTecAD2` data module let me run those three implementations on the same public Can images from MVTec. [MVTec HALCON](https://www.mvtec.com/products/halcon) · [Anomalib project](https://github.com/open-edge-platform/anomalib) · [MVTec AD 2 data module](https://anomalib.readthedocs.io/en/lib-v2.7.1/markdown/guides/reference/data/datamodules/image/mvtecad2.html) · [Experiment report and files](anomalib_testing/ANOMALIB_EXPERIMENT.md)

#### How the runs were set up

The runs used Anomalib `2.7.2.dev0` from source with Python 3.11, PyTorch, and Lightning on Windows. The RTX 5060 Laptop GPU ran in WDDM mode with 8 GiB of VRAM, and the laptop had 32 GB of system RAM. The Can split supplied **412 good training images** and **46 good validation images**, all under regular light. Each of the six test lights had **12 good and 15 defective captures**, for **72 good and 90 defective captures** in total. The lights were regular, overexposed, underexposed, and Shifts 1–3.

All three runs used the default model settings and scored each photo independently. The preprocessor resized the native 2,232 × 1,024 images to **558 × 256 pixels**, preserving the aspect ratio and reducing each axis by a factor of four. Anomalib takes the size as `(height, width)`, so the reported runs use `(256, 558)`. The first attempt used `(560, 256)`, which produced a portrait-shaped input and was discarded. The runs used antialiased resizing, ImageNet normalization, training batches of 8, evaluation batches of 16, and 8 data-loader workers.

| Model | Backbone and feature layers | Fit | Predict |
| --- | --- | ---: | ---: |
| PatchCore | Wide ResNet-50-2, layers 2–3, 10% coreset | 12,155 s (about 3 h 23 min) | 85 s |
| PaDiM | ResNet-18, layers 1–3, 100 retained features per position | 58 s | 60 s |
| STFPM | ResNet-18, layers 1–3, student–teacher checkpoint at epoch 130 | About 2 h 6 min | 65 s |

PatchCore and PaDiM fitted their references in one epoch. Most of PatchCore’s fit time went into k-center greedy selection of **92,287 reference patches**, at about 7.8 selections per second. STFPM trained a student network. In this environment `Engine()` picked up Lightning’s 1,000-epoch default, so I stopped training and scored the saved checkpoint at epoch 130 (global step 6,812). Each STFPM epoch wrote a checkpoint of about 32 MiB. The project evaluator took another one to two minutes per model. These are timings for the recorded stages on this laptop. [Anomalib source build](https://github.com/open-edge-platform/anomalib/tree/3fd9805e1de959c13ad93a24f3f6fae9a7e920a4)

#### How the predictions were scored

1. The adapter read the fields on each `Engine.predict` batch object and turned them into per-image records. It matched predictions by their full dataset-relative paths because good and defective captures can share a filename. All three runs matched **162 of 162 captures**, with zero unmatched predictions.
2. The project evaluator received the continuous `anomaly_map` returned by Anomalib. The default post-processor stayed enabled. For each defective capture the adapter read the annotation as a 2D grayscale mask and resized the score map bilinearly to the native mask shape, 1,024 × 2,232.
3. The evaluator pooled background pixels within each light and weighted each eight-connected defect region equally. Its 4,096-bin threshold sweep integrated overlap up to a 5% pixel false-positive rate and divided the area by 0.05 to get normalized AU-PRO. The reported means weight the included lighting groups equally. Tiny-region overlap measures coverage at the 5% limit for regions occupying at most 0.01% of the source image. The evaluator’s default cutoff of 0.0 applies to the part-decision fields. [Metric implementation](local_contrast/evaluation.py) · [Per-light aggregation](local_contrast/benchmark.py)

The adapter resized maps when a defect mask was present. The 72 good captures have no masks, so their maps contributed background pixels at 256 × 558. Each good capture in the native ensemble contributed sixteen times that pixel count. This changes the weighting of good-image background in the pooled false-positive rate. The scores below describe the recorded runs. A comparison on a common scoring grid would measure how much that weighting mattered.

#### What scored best

| Public Can configuration | AU-PRO, all six | AU-PRO, five changed lights | Regular-light AU-PRO | Worst-light AU-PRO | Tiny overlap, all six |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three banks, six aligned photos per part | **67.53%** | **67.53%** | **67.53%** | **67.53%** | **81.82%** |
| Three banks, one photo per prediction | **66.71%** | **66.62%** | **67.16%** | **65.26%** | **81.64%** |
| Anomalib PatchCore, one resized photo | 7.30% | 6.83% | ≈9.7% | 4.42% | 9.09% |
| Anomalib PaDiM, one resized photo | 2.20% | 2.14% | ≈2.5% | 1.78% | 4.69% |
| Anomalib STFPM, one resized photo | 2.09% | 2.08% | ≈2.1% | 1.95% | 4.59% |

PatchCore led the Anomalib runs. The single-photo ensemble scored **59.41 percentage points higher** than PatchCore across all six lights and **59.79 points higher** across the five changed lights, using the displayed scores. The Anomalib regular-light estimates come from subtracting five times the rounded changed-light mean from six times the rounded all-light mean. The ensemble’s regular-light scores come directly from the saved results. The six-photo version averages the six views into one map per scene using the same banks (seeds 42, 7, and 99 and a local-contrast denominator offset of 1.0). [Saved ensemble results](results/can_ensemble_multiview.json)

#### Why resolution was worth checking

An audit of the 90 defective masks found **108 four-connected regions** of at least four pixels. The median area was **100.5 pixels** (0.0044% of a frame), and the median equivalent diameter was **11.3 pixels**. The 10th to 90th percentile range was 9–409 pixels in area and 3.4–22.8 pixels in diameter. The largest region was 446 pixels. About **28%** of regions were smaller than 8 pixels in diameter, **72%** were smaller than 16, and all were smaller than 32.

At the resized input, PatchCore’s 32 × 70 feature grid represents roughly **32 × 32 native pixels per location**. The finest 64 × 139 feature grid in PaDiM and STFPM represents about **16 × 16**. The median defect is narrower than either footprint. Upsampling lines up the heatmap with the mask after the input resize has already removed fine detail. The native Can model keeps the original detail with overlapping tiles and adds a **25 × 25 local dark-contrast signal** alongside its deep features.

The default Anomalib runs also differed from the Can ensemble in how they handled lighting. They saw regular light during fitting and processed the five changed lights one photo at a time. The ensemble was tuned on these public scenes, and its six-photo version can reinforce responses that stay in the same place across views. Input resolution, feature choice, tuning, view count, and the scoring-grid difference all matter when reading the gap. The earlier full-frame versus tile test is the controlled experiment for the effect of resizing. [Resolution test](#the-first-can-resolution-test) · [Six-view experiment](#12-one-can-under-six-lights)

#### What a native-resolution run would cost

A full-frame attempt at native size with a batch of 32 ran out of GPU memory. At 558 × 256, the 32 × 70 PatchCore feature grid across 412 training images gives **922,880 embeddings** of 1,536 float32 features. That is roughly **5.67 GB before coreset selection** and **0.57 GB after**. At native size the grid would sample every eight pixels, giving 128 × 279 locations per photo and about **90.4 GB** of embeddings before selection. For PaDiM, a single native covariance tensor across **142,848 positions** and **100 retained features** would take about **5.71 GB**, before storing training embeddings or inverting the covariance matrices. STFPM trains a student network directly, with **16 times as many input pixels** per pass at full size.

##### Estimated native-resolution AU-PRO

These ranges are untested projections from the experiment notes for full-size inputs at `image_size=(1024, 2232)` (height, width). They assume the same model settings, regular-light training, and one photo per prediction.

| Anomalib model | Measured at 558 × 256 (all six / five changed) | Projected at native size (all six) | Projected at native size (five changed) |
| --- | ---: | ---: | ---: |
| PatchCore | 7.30% / 6.83% | **≈20–35%** | **≈15–28%** |
| PaDiM | 2.20% / 2.14% | **≈8–18%** | **≈6–14%** |
| STFPM | 2.09% / 2.08% | **≈15–30%** | **≈10–25%** |

The notes also considered an optimistic PatchCore case of **35–40% AU-PRO across all six lights**. That would still be roughly 30 percentage points below the **67.53%** measured for the ensemble using six aligned photos per part.

## Reproduce the study

The experiments ran with Python 3.11, PyTorch 2.11, and an RTX 5060 Laptop GPU with 8 GiB of VRAM. Use MVTec’s download form to get AD 2, then extract it into `data/mvtec_ad_2/` so the Can files are under `data/mvtec_ad_2/can/`. The earlier Bottle check uses the original MVTec AD under `data/mvtec_ad/`. Both datasets are available for noncommercial use under CC BY-NC-SA 4.0. [MVTec AD 2 download and terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) · [Original MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

### Rebuild the three-bank model and six-view figure

From the repository root, run the commands below to fit the three banks, evaluate both input settings, and regenerate the figure at the top. PyTorch downloads the public ResNet-18 weights on first use if they are missing from the local cache.

```sh
python -m pip install -r requirements.txt
python -m local_contrast.multiview --skip-inp
python -m local_contrast.showcase --scene-maps artifacts/can_multiview/can_003.npz --out artifacts/can_multiview/can_ensemble_views.png
```

The evaluation saves `public_can.json` with the per-light results and `can_003.npz` with the maps for the example in `artifacts/can_multiview/`. Remove `--skip-inp` to include the within-image correction experiments.

### Where I put the code

| Files | What they do |
| --- | --- |
| [Can inspector and evaluation](local_contrast/) | Fit the normal reference banks, match scenes and masks, score predictions, and render heatmaps. |
| [Baseline experiments](experiments/baselines/) | Reproduce the native inspector, released CAT and SuperADD runs, and the studies on decision rules, lighting, and adaptation. |
| [Detail experiments](experiments/detail/) | Test pixel-level detail signals, feature banks, and synthetic defects. |
| [Refinement experiments](experiments/refinements/) | Reproduce the seed sweeps, transfer studies, and map and decision experiments. |
| [Anomalib comparison](anomalib_testing/) | Run the Anomalib models and find the recorded logs, checkpoints, and preview images. |
| [Measurements and figures](results/) | Saved reports, per-light scores, and examples from actual runs. |

### Run the earlier inspectors and experiments

These commands reproduce the single-bank Can model from [the local dark-contrast experiments](#11-the-can-scratch-that-kept-me-working).

```sh
python -m local_contrast.inspect_can --image data/mvtec_ad_2/can/test_public/bad/003_shift_1.png --out artifacts/local_contrast/single
python -m local_contrast.inspect_can --public --out artifacts/local_contrast/public
python -m local_contrast.showcase --run artifacts/local_contrast/public/public_can.json --out artifacts/local_contrast/public/per_light.png
```

The single-image command fits the normal references and writes the native-size score map, a preview and a JSON decision. On the illustrated capture the peak was **2.960** against a part threshold of **2.573**, giving **reject**. The public run writes metrics for each light, one decision per capture and the illustrated defect crop. Both commands choose CUDA when available and accept `--device cpu`. The experimental runners under `local_contrast/` retain their own detector and evaluator so the earlier experiments can still be reproduced with their original settings.

Run the earlier native-fusion, scoring and transfer experiments from the repository root.

```sh
python -m experiments.baselines.multiscale_experiment can --previews

python -m experiments.baselines.inspection score can data/mvtec_ad_2/can/test_public/good/000_regular.png --method fusion --output artifacts/inspection

python -m experiments.baselines.multiscale_experiment can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts --previews

python -m experiments.baselines.report_experiments
```

The Can alternatives and dataset replays also run from the repository root. They keep the outputs under their treatment-specific folders.

```sh
python -m local_contrast.run_experiment bottle
python -m local_contrast.run_experiment can
python -m local_contrast.run_experiment can --mode tiled
python -m local_contrast.run_experiment can --augment-lighting
python -m experiments.baselines.calibrate_lighting artifacts/can/full

python -m experiments.baselines.multiscale_experiment can --treatment full
python -m experiments.baselines.multiscale_experiment can --treatment luma
python -m experiments.baselines.multiscale_experiment can --treatment retinex
python -m experiments.baselines.multiscale_experiment can --treatment augmented
python -m experiments.baselines.multiscale_experiment can fabric wallplugs --treatment coreset
python -m experiments.baselines.multiscale_experiment can --backbone gradient
python -m experiments.baselines.multiscale_experiment can --backbone synthetic
python -m experiments.baselines.multiscale_experiment can --backbone distillation

python -m experiments.baselines.compare_scoring can
python -m experiments.baselines.compare_scoring sheet_metal
python -m local_contrast.run_experiment fabric --mode tiled --compare-part-scores
python -m local_contrast.run_experiment wallplugs --mode tiled --compare-part-scores

python -m experiments.baselines.lighting_study can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
python -m experiments.baselines.paired_robustness artifacts/multiscale/native/can
python -m experiments.baselines.support_adaptation artifacts/multiscale/native/can --lighting-audit artifacts/lighting_study/can
python -m experiments.baselines.feature_adaptation
```

The commands below inspect the two parts used in the follow-up figures. Each writes the result to a local directory. The notebook screenshot uses the first command with a different output directory.

```sh
python -m experiments.baselines.inspection score can data/mvtec_ad_2/can/test_public/bad/003_shift_1.png --method position --output artifacts/can_demo
python -m experiments.baselines.inspection regions sheet_metal data/mvtec_ad_2/sheet_metal/test_public/bad/004_shift_1.png --output artifacts/sheet_demo
```

The follow-up experiment log includes the full sequence for rebuilding the region scores, offset and position comparisons, CAT verification, and figures. [Reproduce the follow-up](results/hotspot_study.md#reproduce)

The ConvNeXt experiment uses `timm==1.0.30`, `safetensors==0.7.0`, and the pinned DINOv3 checkpoint saved to the path below. The checkpoint SHA-256 is `a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f`. [Pinned ConvNeXt-Tiny weights](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m/blob/39c7bb32f83c11bbb13cb23058f8852a4dac88ed/model.safetensors)

```sh
python -m pip install timm==1.0.30 safetensors==0.7.0
python -m experiments.baselines.multiscale_experiment can --backbone convnext --weights artifacts/backbones/convnext_tiny_dinov3.safetensors
```

The released-model adapters have their own pinned-source and checkpoint instructions. [CAT commands](results/released_comparators.md#reproduce-the-cat-run) · [SuperADD commands](results/superadd.md)

The default training budget is 96 normal images per category. Each run saves its selected training paths, validation thresholds, per-image decisions, per-light metrics, and optional previews under `artifacts`. All-source fits use `--train-limit 0`. The original gallery comes from `make_project_visuals.py`. The follow-up uses `plot_hotspots.py` and an actual inspector run captured in a local notebook. [Original visual manifest](results/visuals/manifest.json) · [Follow-up examples and selection rules](results/hotspot_study.md)

Once the experiment folders are present, regenerate the charts and inspection figures with Matplotlib 3.10.8 and the checked-in figure script.

```sh
python -m pip install matplotlib==3.10.8
python -m experiments.baselines.report_experiments --chart results/multiscale.png
python -m experiments.baselines.make_project_visuals
```

The private export includes native-resolution maps and binary masks for all **4,090 private images**. The local 19.88 GiB archive passed MVTec’s supplied format checker. It contains 2,045 float16 score maps and 2,045 binary masks for each private split. The archive is a local engineering artifact, with its configuration and SHA-256 recorded in the repository. [Archive manifest](results/benchmark_export.json)

### Anomalib run notes

Use the Anomalib source build listed in the comparison and place the dataset under `data/mvtec_ad_2/`. Run these commands from the repository root. The fit runner includes a pandas 3 compatibility fix for missing mask paths on good samples. New runs save their outputs under `artifacts/anomalib_testing/`. [Anomalib setup and settings](#13-testing-anomalib-on-can) · [Recorded logs](anomalib_testing/logs/)

```sh
python -m anomalib_testing.research_extra12
python -m anomalib_testing.research_stfpm_eval
```

The first command fits all three models. In the recorded run, STFPM training was stopped after saving epoch 130. The second command works on its own and scores that saved checkpoint. The three recorded checkpoints use Git LFS. If the checkout has pointer files, run `git lfs pull` before loading the weights. The `all6`, `changed5`, `worst`, and `tiny` fields in the original logs are the values used above. [Recorded checkpoints and previews](anomalib_testing/results/) · [STFPM checkpoint](anomalib_testing/results/Stfpm/MVTecAD2/can/v0/weights/lightning/model.ckpt)

## What I learned

The controlled comparison shows that retaining native image detail can substantially improve small-defect localization. Combining feature scales improved tiny-region coverage across the categories in the study. The photometric, scoring, and adaptation experiments showed how easily a promising Can result could fail to transfer or move the pass/fail trade-off.

Several limits matter when interpreting the numbers.

- The public Can sample has 15 defective and 12 good scenes. Repeated lighting views give paired evidence, while the number of independent scenes remains small.
- I used public results to guide the sequence of experiments. The final tables describe an exploratory study of that public split.
- Some tiny-region groups contain only one to four qualifying components per light, so the category mean gives sparse groups substantial weight. The low-FPR localization metric sweeps thresholds. The actual part decisions use fixed normal-validation thresholds, so the two measurements answer different questions.
- The histogram AU-PRO approximates an exact threshold sweep. The 5% integration limit must match before comparing it with another reported AU-PRO value.
- The ten configurations share source budgets where possible. The released models have their own architectures, references, training, and binary postprocessing.
- The recorded latency boundaries and workload vary by runner. The final local comparison reports the measured times on the same laptop.
- Selected visual examples illustrate individual behavior. The tables contain the full evaluated sets.

On another product, the first thing I’d check is where the detector looked on a missed part. Drawing five small boxes told me more about the next experiment than a single average score did. On Sheet Metal that check led to 20 additional detections. On Can it explained why changing the score alone had so little chance of recovering many of the missed locations. Going back to the native pixels gave the Can model a better map. Three reference banks improved it again, and six aligned views gave it one more gain. Tracking the part decisions alongside the heatmaps showed where those gains stopped.

## Data, images, and references

The images and ground-truth masks shown here come from MVTec AD and MVTec AD 2. The displayed heatmaps are outputs of the project models on those images. MVTec datasets are licensed under CC BY-NC-SA 4.0. The figure records retain source image paths, scene identifiers, and rendering choices. [MVTec AD license and download](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [MVTec AD 2 license and download](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

The accompanying research notes and result reports link the papers, pinned implementations, and machine-readable measurements used throughout the study. [Research review](docs/research.md) · [Per-configuration data](results/multiscale.json) · [Released-model data](results/released_comparators.json)
