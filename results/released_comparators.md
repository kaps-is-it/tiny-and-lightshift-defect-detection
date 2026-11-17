# Released models on the same public Can scenes

Native RGB fusion, the released SuperADD configuration, and the released CAT checkpoint scored the same 162 captures of 27 Can scenes. The table covers the five changed lights, with 75 defective captures and 60 good captures. Localization uses the same native-coordinate evaluator and equally weights each light. Tiny overlap covers mask regions at most 0.01% of the source image, measured at 5% pixel false positives.

| Can model | AU-PRO at 5% | Tiny overlap at 5% FPR | Pixel F1 | Missed defects | False rejects | Median compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native RGB fusion | 46.39% | 73.14% | 2.89% | 54 / 75 | 15 / 60 | 215 ms |
| CAT released checkpoint | 33.87% | 42.00% | 4.21% | 27 / 75 | 32 / 60 | 2027 ms |
| SuperADD released bank | 4.91% | 9.04% | 0.00% | 34 / 75 | 28 / 60 | 3141 ms |

Native RGB fusion learns its score threshold from good regular-light validation images. SuperADD uses its saved threshold and mask postprocessing. CAT uses the authors' fixed Can threshold and connected-component cleanup. Part decisions for both released methods reject a capture if the resulting mask contains a positive pixel. Each row therefore measures an operating point specific to that method. The CAT run also reports a separate part rule fitted on the 46 good validation images. The machine-readable comparison includes all six lights and intervals for changed-light error differences bootstrapped by dataset scene. [Comparison data](released_comparators.json) · [CAT per-light results](cat_release.json) · [SuperADD per-light results](superadd_release.json)

Native RGB fusion had the strongest low-FPR localization and covered more of the tiny labeled regions. CAT’s published mask rule had the highest pixel F1. It caught 27 more defective captures than native fusion and rejected 17 more good captures. The separate normal-validation CAT rule missed 24 of 75 defective captures and falsely rejected 38 of 60 good ones. SuperADD gave the highest mean part AUROC at 60.4%, while its released masks had zero pixel F1. The three pipelines therefore answered the ranking, localization, and pass/fail questions differently.

At the released 0.625× input scale, SuperADD’s 16-pixel feature stride spans about 26 native pixels. The fine branch of native fusion scores features on a four-pixel grid. The cross-model comparison changes architecture, spatial sampling, and reference construction together. The matched ResNet control compares native tiling with full-frame resizing at the same backbone and normal source budget. [Controlled resolution comparison](experiments.md#can-method-screen)

## Reproduce the CAT run

Use the Python environment from the [README](../README.md#reproduce-the-study). Clone the author's repository to the path expected by the runner and check out the tested revision. [Author code and checkpoints](https://github.com/VisualScienceLab-KHU/VAND2026_CAT/tree/9038ee3dddf947f30c61251c6fce898f2334ba6c)

```sh
git clone https://github.com/VisualScienceLab-KHU/VAND2026_CAT.git artifacts/upstream-cat/VAND2026_CAT-9038ee3dddf947f30c61251c6fce898f2334ba6c
git -C artifacts/upstream-cat/VAND2026_CAT-9038ee3dddf947f30c61251c6fce898f2334ba6c checkout 9038ee3dddf947f30c61251c6fce898f2334ba6c
```

Extract the Can `model.pth` from the authors' archive under the `CLASS=can` run. Save it as `artifacts/upstream-cat/checkpoints/can-model.pth`. The SHA-256 is `f2a9b45d4dc65779cc4dc15e2972de17fdb268982f6e389e7bc942df8894493e`. The checkpoint already includes the DINOv3 ViT-L/16 encoder. [Author checkpoint archive](https://drive.google.com/file/d/1mBYLvNc4sktJodrzwkjx7kUjOr9zH8jF/view)

The runner preserves the four-by-four split, 448-pixel input, cosine residual and uint8 tile quantization from the author code. The published Can mask rule thresholds the stitched scores strictly above 42 and removes 8-connected components below the larger of eight pixels or 0.002% of image area, rounded down. The authors document calibration on good validation data. [Pinned CAT code and threshold protocol](https://github.com/VisualScienceLab-KHU/VAND2026_CAT/tree/9038ee3dddf947f30c61251c6fce898f2334ba6c)

```sh
python run_cat.py --device cuda --batch-size 4 --smoke
python run_cat.py --device cuda --batch-size 4 --output artifacts/cat/can-release
python multiscale_experiment.py can --output artifacts/release-comparison
```

The CPU and CUDA CAT smoke tests produced the same reported metrics. The full comparison ran on an RTX 5060 Laptop GPU. The sibling report documents the released SuperADD run, including the timm backbone conversion and local assets. [SuperADD reproduction](superadd.md)
