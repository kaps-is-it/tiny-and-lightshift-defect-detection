# Detecting Tiny Industrial Defects Under Changing Light

## Research question

Can an inspector catch more tiny defects by processing overlapping tiles of the original image, and can a few verified-good reference parts keep its pass/fail decisions reliable after the lighting changes?

A scratch can become too small to distinguish when a high-resolution photo is shrunk to fit the model. Glare or shadows can also make a good part look damaged or hide a real defect before the model sees the image. First compare full-frame resizing and tiling under regular lighting. Then repeat the comparison on the same scenes under changed lighting and test whether recalibration helps. [MVTec AD 2 paper, Sections 3.2–3.4 and 4.3.3](https://link.springer.com/article/10.1007/s11263-026-02743-0)

## Datasets we will use

Keep both downloads. The original MVTec AD is a quick way to check that training, thresholding, mask alignment, and evaluation all work. MVTec AD 2 provides the matched lighting captures for the research experiment and the private splits for the leaderboard. Fit a separate reference bank per category from that benchmark's good training images. [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

| Local folder | Inventory | Role |
| --- | --- | --- |
| `MVTECH_ANAMOLY_DATASET` | Original MVTec AD. 15 categories, 5,354 inspection images and 1,258 masks. | Use `bottle` for the first end-to-end run. Save the other categories for later checks if needed. |
| `mvtec_ad_2_dataset` | MVTec AD 2. 8 categories, 8,004 inspection images and 705 public masks. | Start with `can`, check transfer on `sheet_metal`, and run the full dataset for official evaluation. |

MVTec AD 2 has 2,528 good training images, 302 good validation images, and 1,084 labeled public test images. Each private split contains 2,045 images. The local audit found a matching mask of the right dimensions for every labeled defective image in both benchmarks.

| AD 2 category | Good public scenes | Defective public scenes | Lighting conditions per scene |
| --- | ---: | ---: | ---: |
| `can` | 12 | 15 | 6 |
| `fabric` | 11 | 15 | 6 |
| `fruit_jelly` | 5 | 15 | 4 |
| `rice` | 7 | 15 | 6 |
| `sheet_metal` | 4 | 15 | 6 |
| `vial` | 5 | 15 | 7 |
| `wallplugs` | 10 | 15 | 6 |
| `walnuts` | 10 | 15 | 6 |

`can` is the primary category. It has 412 good training images, 46 good validation images, and 162 public captures of 27 independent scenes. Across the 15 defective regular-light scenes, the masks cover 25–450 pixels each in images measuring 2,232 × 1,024. All are under 0.02% of the image. Added spotlights create reflections, so the same scenes test both the effect of shrinking the image and the effect of changed lighting. Use each capture's own mask because lighting can change how much of a defect is visible. [MVTec AD 2 paper, Sections 3.3–3.5](https://link.springer.com/article/10.1007/s11263-026-02743-0)

`sheet_metal` tests whether the approach transfers to long, narrow defects and a very different 4:1 grayscale image. Use it for localization. Its four good public scenes give very coarse false-reject estimates. If we extend the recalibration pilot to a second category, use `fabric`, which has 11 good scenes.

Use the public split for local comparisons. Use the benchmark server to test the final candidate on the private splits. The two private splits use the same scenes, and the mixed set includes both familiar and unfamiliar lighting. Its filenames do not identify the specific lighting setup. Ground truth for both splits stays on the server. [MVTec AD 2 dataset and evaluation](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) · [Published protocol, Section 5](https://link.springer.com/article/10.1007/s11263-026-02743-0)

## Leaderboard targets

These are the highest posted scores for each metric and column in the November 14, 2026 snapshot. Each maximum can come from a different entry. The main table sorts by the eight-category average AU-PRO at a 5% pixel false-positive limit on `test_private`. Our first leaderboard target is the `can` column on `test_private_mixed`, where lighting varies. [Official leaderboard](https://benchmark.mvtec.com/) · [Leaderboard data](https://www.benchmark.mvtec.com/v1/leaderboard)

| Metric | `can` regular | `can` mixed | Eight-category average, regular | Eight-category average, mixed |
| --- | ---: | ---: | ---: | ---: |
| AU-PRO at 5% | 94.68% | 80.97% | 88.72% | 82.44% |
| Segmentation F1 | 25.22% | 14.04% | 65.30% | 61.65% |
| Classification F1 | 75.70% | 73.15% | 85.89% | 84.50% |

The `YL` entry posted 80.97% mixed-light AU-PRO on `can` and 1.00% mixed-light segmentation F1 on the same run. Track both metrics because good anomaly scores and useful binary masks answer different questions. Start by beating our reproduced baseline, then aim above the published CAT entry at 41.46% on the official mixed `can` split. The leaderboard also lists `SuperADD-A-B-Distilled` at 51.41%. The stretch targets are 80.97% on `can` and 82.44% on the eight-category average. Use private server results for these comparisons and refresh the snapshot before making a rank claim. [Official leaderboard](https://benchmark.mvtec.com/) · [CAT implementation](https://github.com/VisualScienceLab-KHU/VAND2026_CAT)

SuperADD already uses a pretrained DINOv3 backbone, overlapping crops, and brightness augmentation. Use the published implementation as the first strong local comparator. Test whether adding a finer-scale crop pass improves thin-scratch localization. Fuse the score maps in original coordinates using scaling estimated from good validation images. Compare the standard configuration, fine-scale pass, and fused map under the same evaluation conditions. Inspect the smallest defect regions, false positives from glare and printed edges, and the extra time per part. Keep `can` as the first milestone and apply the same pipeline to the remaining classes before comparing eight-category averages. [SuperADD report, Sections 2–4](https://arxiv.org/html/2605.14808v1)

## MVTec development utilities

- `mvtec_ad_2_public_offline.py` is the reference for split names, loading images and public masks, and naming output maps. Use a configurable root and `pathlib` in our loader. The supplied root is a placeholder, and its public mask lookup splits on `/bad/`, which fails with ordinary Windows backslash paths.
- `check_and_prepare_data_for_upload.py` and `utils.py` check names, counts, and formats and produce a local `.tar.gz`. The official workflow expects all eight categories and both private splits, totaling 4,090 single-channel `float16` TIFF score maps. The optional parallel tree of binary `0`/`255` PNG masks enables segmentation and classification F1. Check that maps are finite and spatially aligned before packaging. [MVTec submission page](https://benchmark.mvtec.com/submit)
- `measure_runtime_and_memory.py` is a template for measuring inference time and peak reserved GPU memory. For the tile comparison, also time preprocessing, tile inference, and stitching per original image on the same hardware. The bundle's README points to a separate AU-PRO evaluator for the labeled public split. Use a 0.05 pixel false-positive limit for the leaderboard-style metric. [MVTec AD evaluation code page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

Keep the official candidate fixed before running the private splits. Fit its reference bank and thresholds using the designated good training and validation data. Report the few-shot experiment with shifted reference captures as a local commissioning study. A threshold adjustment can change F1 and pass/fail outcomes. Improving continuous AU-PRO requires better anomaly maps. MVTec derives part-level F1 from whether the submitted binary map contains a positive pixel. Report our local image-score decision separately when it uses a different rule. [Published protocol, Sections 4.2 and 5](https://link.springer.com/article/10.1007/s11263-026-02743-0)

## Experiment

PatchCore uses a pretrained CNN to represent small image patches and stores representative patch features from good parts. A test patch receives a high anomaly score when its nearest normal reference is far away. The individual patch scores produce a heatmap, and each image gets a score for the part-level decision. The first runs use the PyTorch baseline in this repo with a uniformly sampled reference bank and raw nearest-neighbor scores. The [README](README.md#results) has the public results and reproduction commands. [PatchCore paper](https://arxiv.org/abs/2106.08265)

The laptop has a GeForce RTX 5060 with about 8 GiB of VRAM. PyTorch sees CUDA in the current environment. Start with ResNet-18, small tile batches, capped reference banks, and chunked distance queries. Verify compatible `torchvision` and `tifffile` in a project environment. Profile SuperADD on one image before committing to a full run. If its published setup exceeds memory, use a smaller backbone or fewer tiles per batch and record any configuration differences. A smaller model is a separate comparator. Stream exports and keep datasets, caches, weights, and generated maps outside version control.

### Test whether more image detail helps

- **Full-frame baseline.** Resize the entire image to fit a 256-pixel input, preserving aspect ratio and padding as needed so the edges remain visible. Exclude padding from scoring and evaluation.
- **Tiled PatchCore.** Start with 256-pixel tiles and 25% overlap. Build the normal reference using tiles from the same good training images. Restore the predicted tile maps to the original coordinates and blend their overlaps. Cover the full frame, including the edges. Set tile size, stride, and blending before evaluating the labeled test set.
- **Compare on both lighting conditions.** Use the same backbone, source training scenes, held-out evaluation scenes, and memory bank budget. Compute a part score from each complete heatmap using the same rule. Give each variant its own threshold fitted by the same validation rule on regular-light good images. Start with regular lighting, then run both variants on every available shifted capture of the same scenes. If resources allow, add a larger full-frame input to see how much of the gain also comes from resizing less.

For each variant, record the input scale, missed tiny defects, false rejects, and time per original image, including tiling and stitching. Inspect cases where the baseline misses a defect that the tiled map locates, as well as seams, glare, and normal texture that get flagged. The comparison tests the whole tiling approach, which changes both effective resolution and the amount of surrounding context. [MVTec AD 2 paper, Sections 4.3.3 and 6.1](https://link.springer.com/article/10.1007/s11263-026-02743-0)

### Test recovery after a lighting change

| Variant | What to implement | Question it answers |
| --- | --- | --- |
| Fixed PatchCore | Fit the normal reference on regular-light training images. Start with the 90th percentile of held-out regular-light good scores as the per-category pass/fail threshold. Use it under every lighting condition. | How much does a fixed inspector degrade after lighting changes? |
| PatchCore with lighting augmentation | Build a reference from normal training images with mild brightness and contrast augmentation. Use the same backbone, preprocessing, and memory budget, then apply the same threshold calibration rule to regular-light validation. | How much robustness can we get before deployment, using only the source data? |
| PatchCore with threshold recalibration | Keep the original detector. When the light changes, score a few verified-good reference scenes under both settings. Add the median paired score difference to the original threshold for that lighting setup. | Can a small set of reference parts reduce false rejects without increasing defect escapes? |

The 90th percentile is an initial operating point for the experiment. Measure the actual false-reject rate under both regular and shifted lighting. When wiring up Anomalib, explicitly configure the calibration split and threshold so framework defaults cannot draw calibration examples from the final test set.

Apply the fixed and recalibrated thresholds to scores from both full-frame and tiled PatchCore. Run the augmentation comparison on the full-frame baseline first.

Reserve four of the 12 good public `can` scenes before scoring. Use two of those four for the smaller calibration budget. All variants score the remaining eight good scenes and all 15 defective scenes, under each lighting condition. Record the seed. One false reject changes the per-lighting rate by 12.5 percentage points, so report counts and treat this as a pilot. Use additional prespecified draws for a sensitivity check if needed.

Paired recalibration assumes that the station has regular-light captures of its known-good reference parts and can photograph them again under the new setup. A threshold change can improve the pass/fail operating point. A better heatmap or better ranking within a condition requires a change to the underlying scores. Existing work such as SPARC uses verified-good samples for a more complex feature correction. This project studies what the simpler change buys at a factory decision threshold. [SPARC preprint](https://arxiv.org/abs/2608.18585)

### Detect when recalibration is needed

After measuring the three variants above, prototype a lighting monitor using batch-level brightness, contrast, and highlight clipping measured on known-good reference parts. Establish the usual range from regular-light data. In offline replay, compare equal-sized regular and shifted batches of held-out reference scenes. Count false alarms and missed shifts. Use the lighting labels only to score the replay. A shift alarm requests a reference check and can route questionable parts to review. A change in image statistics signals a possible acquisition change. Labeled evaluation establishes whether inspection performance actually degrades.

## Evaluation

1. **Keep the same physical scene together.** Build a manifest with category, split, scene ID, lighting condition, label, image path, and mask path. Public filename counters restart in `good` and `bad`, so group by category, split, label folder, and numeric prefix. The remaining filename suffix gives the lighting condition. Every lighting capture of a calibration scene belongs outside the scored evaluation set. Freeze the model configurations and evaluation rules before looking at the resulting test scores.
2. **Estimate the lighting penalty within MVTec AD 2.** Compare regular and changed lighting on matched held-out parts, and show results per lighting condition. The two MVTec benchmarks cover different products, so their score difference cannot isolate the effect of lighting.
3. **Count false rejects and missed defects.** Report counts and rates by lighting condition, including recall on `can` scenes whose regular-light masks contain only tiny regions. Define tiny as an 8-connected region covering at most 0.01% of the original image. This gives 11 tiny regions out of 16 and 10 tiny-only scenes out of 15 on regular-light `can`. Carry the regular-light scene IDs across conditions. Report denominators and use paired, scene-level uncertainty estimates when the sample supports them.
4. **Check whether the heatmap finds the actual defect.** A part can be rejected because of a larger defect even when the tiny defect was missed. Report AU-PRO at a 5% false-positive-pixel limit, segmentation F1, and average overlap for tiny regions at a matched pixel false-positive rate. For region-level scoring, apply the same area cutoff to each capture's own mask and report how many regions qualify in that capture or condition. Evaluate maps in the original image coordinates. Flag cases where the defect has little visible contrast even at the original resolution. [MVTec AD 2 paper, Sections 3.4 and 4.2](https://link.springer.com/article/10.1007/s11263-026-02743-0)
5. **Evaluate review only if it is implemented.** Compare policies at the same review budget. Report the share of parts reviewed and count defective parts automatically passed out of all defective parts. Show the review volume alongside any reduction in escapes.

Report how much tiling changes tiny-defect detection and localization, what it costs in processing time, and whether the gain holds under changed lighting. Show what recalibration does to both false rejects and defect escapes. If the public sample is too small to distinguish improvements, give the counts and show the paired cases. Log input size, runtime, hardware, split seeds, and exact configs.

## Build order and portfolio deliverables

1. **Build the data and evaluation pipeline.** Turn the verified layout into a repeatable manifest. Add checks for missing files, image/mask shape, repeated scene IDs, and all 16 private category/split folders. Fix the public `can` scene groups before scoring.
2. **Run `bottle` end to end.** From 209 good training images, hold out 42 at a fixed seed for calibration and fit PatchCore on the other 167. Run on the 20 good and 63 defective test images. Save scores, timings, and an overlay aligned with its ground-truth mask.
3. **Measure the resolution and lighting effects on `can`.** Run full-frame and tiled PatchCore on the same scenes under all six conditions. Report tiny-defect localization, false rejects, missed defects, and runtime. Test lighting augmentation as a source-only control.
4. **Challenge a stronger model.** Run SuperADD on the same scenes. Compare its standard map, a fine-scale pass, and their fusion. Measure the gain at the same pixel false-positive rate, record the time cost, and use `sheet_metal` to check whether the improvement transfers to a different surface and aspect ratio.
5. **Test the deployment workflow.** Use the reserved good `can` scenes to compare the fixed threshold with two- and four-scene recalibration. Run the lighting alert as an offline replay once the model comparisons work.
6. **Run the official comparison.** Freeze the selected source-only configuration, generate both private splits for all eight categories, and check the submission archive with MVTec's tool. Use the server's returned results for any leaderboard claim.
7. **Package the study.** Publish runnable code, pinned dependencies, configs, per-condition tables, and a short report. Show the same part under regular light and glare, with an original-resolution crop, both heatmaps, and the decision before and after recalibration. Include a failure case. Keep downloads out of Git and link to the official sources. Both datasets use CC BY-NC-SA 4.0. [MVTec AD license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [MVTec AD 2 license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)

The first build milestone is a verified `bottle` run and the `can` scene manifest. The first research result is the paired full-frame versus tiled comparison, with tiny-defect localization, misses, false rejects, and runtime under all six lighting conditions.
