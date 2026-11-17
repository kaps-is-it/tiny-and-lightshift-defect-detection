# Detecting Tiny Industrial Defects Under Changing Light

A faint scratch can disappear when a camera image is resized for a vision model. A different light can wash out that same scratch or turn a reflection into a false alarm. This project measures both failures on manufactured parts photographed repeatedly under controlled changes in lighting.

The inspector models normal appearance from defect-free images. It scans overlapping crops of each full-resolution capture, scores fine detail and surrounding context, and stitches the scores back into the camera coordinates. The experiments track localization and part decisions separately. A heatmap can point to the right scratch even when an inspection threshold passes the defective part.

## Results

![Shifted-light localization across eight categories and Can ablations](results/multiscale.png)

- **Can localization:** Tiled ResNet fusion reached **46.4% AU-PRO** under changed lighting versus **4.6%** for its matched full-frame control. Tiny-region overlap reached **73.1%** at 5% pixel FPR. Mild Retinex nudged AU-PRO to 46.9% and missed more defective parts at the fixed threshold.
- **Transfer:** Across eight categories, fusion raised mean tiny-region overlap from **63.7% to 73.2%**. Mean AU-PRO moved from **39.7% to 39.9%**. Under the fixed thresholds, fusion missed 448 of 585 shifted defective captures versus 388 for context, while falsely rejecting 43 of 315 good captures versus 63.
- **Adaptation:** Neither median-offset nor bounded affine score calibration jointly reduced misses and false rejects across the 495 Can support sets. An independent acquisition audit found reduced median defect contrast after resizing in all 47 category-light groups.

[Full results, limits, and how to repeat the comparisons](results/experiments.md)

The MVTec AD 2 public split has 1,084 captures of 184 scenes across eight categories. For the main comparison, both branches saw the same 96 evenly spaced good training images per category. Each branch used the good regular-light validation split to set a fixed threshold. Defect masks were used to score the outputs. AU-PRO here is the per-region overlap area up to 5% pixel false positives, estimated with 4,096 score bins. Each lighting group gets equal weight within a category. The private benchmark is evaluated by MVTec. [Paper and benchmark protocol](https://link.springer.com/article/10.1007/s11263-026-02743-0) · [Official leaderboard](https://benchmark.mvtec.com/)

## Reproduce the study

Use Python 3.11 and the matching PyTorch and torchvision builds for your device. The main experiments used CUDA 12.8 on an RTX 5060 Laptop GPU. [PyTorch installation](https://pytorch.org/get-started/previous-versions/#v2110)

```sh
python -m pip install -r requirements.txt
```

Extract MVTec AD 2 into `mvtec_ad_2_dataset`. Extract the original MVTec AD into `MVTECH_ANAMOLY_DATASET` if you want to reproduce the Bottle pipeline check. The programs also accept `--dataset`. Both datasets are available for noncommercial research under CC BY-NC-SA 4.0. [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) · [Original MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

```sh
python run_experiment.py bottle
python multiscale_experiment.py can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts --previews
python multiscale_experiment.py can --treatment full --device cpu
python multiscale_experiment.py can --treatment luma
python multiscale_experiment.py can --treatment retinex
python multiscale_experiment.py can --treatment augmented
python multiscale_experiment.py can --treatment coreset
python multiscale_experiment.py can --backbone gradient
python multiscale_experiment.py can --backbone synthetic
python multiscale_experiment.py can --backbone distillation
python report_experiments.py
```

For the DINOv3 ConvNeXt experiment, install `timm==1.0.30` and `safetensors==0.7.0`. Download the pinned checkpoint and pass its path with `--weights`. Its SHA-256 is `a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f`. To render the summary chart, install `matplotlib==3.10.8` and run the report with `--chart results/multiscale.png`. [Pinned DINOv3 ConvNeXt-Tiny checkpoint](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m/blob/39c7bb32f83c11bbb13cb23058f8852a4dac88ed/model.safetensors)

```sh
python multiscale_experiment.py can --backbone convnext --weights artifacts/backbones/convnext_tiny_dinov3.safetensors
python lighting_study.py can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
python paired_robustness.py artifacts/multiscale/native/can
python support_adaptation.py artifacts/multiscale/native/can --lighting-audit artifacts/lighting_study/can
```

The experiment writes per-image decisions, normal validation scores, and per-light metrics under `artifacts/multiscale`. Model weights and data stay outside Git. Set `--train-limit 0` to use every good training image. Source dependence, few-part adaptation, and extra-view analysis are described in the [acquisition and adaptation report](results/acquisition_and_adaptation.md).

## Score images and export for MVTec

Fit a category reference once per invocation, score a batch of images, and save inspection previews and decisions. The optional review flag defers images where the fine and context decisions disagree.

```sh
python inspection.py score can mvtec_ad_2_dataset/can/test_public/good/000_regular.png --method fusion --output artifacts/inspection
```

The benchmark command fits each category using its good train and validation images, then generates continuous float16 TIFF maps for both private splits. By default the maps retain native image dimensions. `--with-masks` adds binary maps made with the normal-validation threshold. `--resume` continues the same settings after an interrupted run. Add `--checker MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py` to also run the organizer's local checker before packaging. [Official submission instructions](https://benchmark.mvtec.com/submit)

```sh
python inspection.py benchmark --method fusion --with-masks --output artifacts/submission --archive artifacts/submission.tar.gz
```

## Research notes

- [Full experimental results](results/experiments.md)
- [Acquisition and adaptation](results/acquisition_and_adaptation.md)
- [Literature review and experimental rationale](RESEARCH.md)
- [Original protocol and MVTec dataset inventory](PROJECT_PLAN.md)
- [Earlier part-scoring comparison](results/scoring_comparison.md), [public transfer pilot](results/hybrid_transfer.md), and [reduced SuperADD runs](results/superadd.md)
