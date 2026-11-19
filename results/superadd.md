# SuperADD on the public can scenes

## Released Can reference bank

The released configuration was evaluated on all 162 public Can captures from 27 scenes. It uses 100,000 reference embeddings at each of four DINOv3 ViT-H+/16 layers, a 0.625× input scale, 640-pixel windows, and the author's saved threshold and mask postprocessing. The laptop runner uses the timm conversion of the backbone with the original rotary-period precision and bounds the memory used by the nearest-neighbor query. [Author implementation and bank](https://github.com/LukasRoom/SuperADD#reproduce-our-results) · [Converted backbone](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m)

| Captures | AU-PRO at 5% | Tiny-region overlap | Mean part AUROC | Misses from binary masks | False rejects from binary masks |
| --- | ---: | ---: | ---: | ---: | ---: |
| All six lights | 4.95% | 9.05% | 60.56% | 45 / 90 | 29 / 72 |
| Five shifted lights | 4.91% | 9.04% | 60.44% | 34 / 75 | 28 / 60 |

The released binary masks had zero pixel F1 in all six public groups. Median per-light inference was about 3.14 seconds, with 5,848 MiB peak CUDA allocation. These are local measurements using the public labels and this project's histogram evaluator. [Per-light release metrics and asset hashes](superadd_release.json)

Use the pinned checkout and dependencies below. Put the authors' `can.npz` and its matching `can.json` in `artifacts/superadd-assets`. Save the ViT-H+/16 timm checkpoint there as `dinov3-vith16plus-timm.safetensors`. Asset hashes are in the per-light results. [Released reference banks](https://owncloud.fraunhofer.de/index.php/s/THkX7W8AhRd2RCs) · [ViT-H+/16 model files](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m/tree/main)

```sh
python -m experiments.baselines.run_superadd --output artifacts/superadd/can-release
```

## Reduced ViT-S comparison

The reduced runs use the authors' SuperADD implementation on the same public Can scenes. They use a public DINOv3 ViT-S/16 checkpoint and a smaller normal reference bank. It keeps SuperADD's patch extraction, reference selection, distance scoring, and binary-mask postprocessing. [SuperADD report](https://arxiv.org/abs/2605.14808) · [Author implementation](https://github.com/LukasRoom/SuperADD) · [DINOv3 checkpoint](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m)

The reference uses every 16th image from the sorted `can/train/good` split. SuperADD uses 22 of those images to fit the bank and four to set its pixel threshold. The run uses 384-pixel windows, a 64-pixel half-overlap, four transformer layers, and at most 12,000 reference vectors per layer. The 162 public captures and their pixel masks are used for evaluation.

| Model | Mean AU-PRO at 5% | Tiny-region overlap at 5% FPR | Mean part AUROC | Pooled misses | Pooled false rejects | Median image time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reduced SuperADD, 0.625× image scale | 6.4% | 17.2% | 49.4% | 68 / 90 | 13 / 72 | 345 ms |
| Reduced SuperADD, native image scale | 11.1% | 9.1% | 52.1% | 67 / 90 | 12 / 72 | 636 ms |

Native-scale input improved overall AU-PRO and worsened the overlap on tiny regions. It also took about 1.8 times as long per capture. Both runs had zero pixel segmentation F1 at the fitted thresholds. The tiled PatchCore baseline in that comparison reached 39.2% AU-PRO. See the [expanded study](experiments.md) for the later comparisons.

The binary-mask decisions above follow SuperADD's postprocessed mask. Part AUROC uses the peak continuous score. Each run fits its bank and threshold at its own input scale. Timings include preprocessing and restoration to the original frame on the RTX 5060 Laptop GPU. Image decoding sits outside that timing. SuperADD builds its feature map from whole 16-pixel tokens. At native scale, the upstream token grid covers 2,224 of the 2,232 image columns and the returned map is stretched to the frame width.

## Reproduce the reduced run

Start from the Python environment in the [README](../README.md#reproduce-the-study) with CUDA available. Install the optional packages and check out the upstream revision used here.

```sh
python -m pip install timm==1.0.30 safetensors==0.7.0 opencv-python==4.13.0.92 tqdm
git clone https://github.com/LukasRoom/SuperADD.git artifacts/upstream-superadd
git -C artifacts/upstream-superadd checkout 44cf25144442fbbc1334ea59d1632327a4376d1a
```

Download `model.safetensors` from the timm model card to `artifacts/superadd-assets/dinov3-vits16-timm.safetensors`. The runner also accepts an explicit `--weights` path. [Model files](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m/tree/main)

```sh
python -m experiments.baselines.run_superadd --train-small
python -m experiments.baselines.run_superadd --train-small --native
```

The runs write `metrics.json` and `predictions.csv` under `artifacts/superadd/can-small/` and `artifacts/superadd/can-small-native/`.
