# SuperADD on the public can scenes

This experiment runs the authors' SuperADD implementation on the same 27 public `can` scenes used in our PatchCore experiments. The laptop run uses a public DINOv3 ViT-S/16 checkpoint and a smaller normal reference bank. It keeps SuperADD's patch extraction, reference selection, distance scoring, and binary-mask postprocessing. [SuperADD report](https://arxiv.org/abs/2605.14808) · [Author implementation](https://github.com/LukasRoom/SuperADD) · [DINOv3 checkpoint](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m)

The reference uses every 16th image from the sorted `can/train/good` split. SuperADD uses 22 of those images to fit the bank and four to set its pixel threshold. The run uses 384-pixel windows, a 64-pixel half-overlap, four transformer layers, and at most 12,000 reference vectors per layer. The 162 public captures and their pixel masks are used for evaluation.

| Model | Mean AU-PRO at 5% | Tiny-region overlap at 5% FPR | Mean part AUROC | Pooled misses | Pooled false rejects | Median image time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reduced SuperADD, 0.625× image scale | 6.4% | 17.2% | 49.4% | 68 / 90 | 13 / 72 | 345 ms |
| Reduced SuperADD, native image scale | 11.1% | 9.1% | 52.1% | 67 / 90 | 12 / 72 | 636 ms |

Native-scale input improved overall AU-PRO and worsened the overlap on tiny regions. It also took about 1.8 times as long per capture. Both runs had zero pixel segmentation F1 at the fitted thresholds. The best local AU-PRO on these scenes remains the tiled PatchCore baseline at 39.2%.

The binary-mask decisions above follow SuperADD's postprocessed mask. Part AUROC uses the peak continuous score. Each run fits its bank and threshold at its own input scale. Timings include preprocessing and restoration to the original frame on the RTX 5060 Laptop GPU. Image decoding sits outside that timing. SuperADD builds its feature map from whole 16-pixel tokens. At native scale, the upstream token grid covers 2,224 of the 2,232 image columns and the returned map is stretched to the frame width.

The published ViT-H+ configuration uses a larger released bank. Its 2 GB download did not finish during this experiment, so these results cover the reduced ViT-S/16 configuration. [Model configuration and assets](https://github.com/LukasRoom/SuperADD#reproduce-our-results)

## Reproduce the reduced run

Start from the Python environment in the [README](../../README.md#reproduce-the-study) with CUDA available. Install the optional packages and check out the upstream revision used here.

```sh
python -m pip install timm==1.0.30 safetensors==0.7.0 opencv-python==4.13.0.92 tqdm
git clone https://github.com/LukasRoom/SuperADD.git artifacts/upstream-superadd
git -C artifacts/upstream-superadd checkout 44cf25144442fbbc1334ea59d1632327a4376d1a
```

Download `model.safetensors` from the timm model card to `artifacts/superadd-assets/dinov3-vits16-timm.safetensors`. The runner also accepts an explicit `--weights` path. [Model files](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m/tree/main)

```sh
python local_contrast/run_superadd.py --train-small
python local_contrast/run_superadd.py --train-small --native
```

The runs write `metrics.json` and `predictions.csv` under `artifacts/superadd/can-small/` and `artifacts/superadd/can-small-native/`.
