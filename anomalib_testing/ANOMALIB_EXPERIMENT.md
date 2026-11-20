# Anomalib on public MVTec AD 2 Can

## Results in brief

The three Anomalib models ran on the same Can public split. The project evaluator scored their continuous maps and the Can ensemble maps. The Anomalib runs used 558 × 256 inputs and scored one photo at a time.

| Method | AU-PRO, all six lights | AU-PRO, five changed lights | Worst-light AU-PRO | Tiny overlap, all six |
| --- | ---: | ---: | ---: | ---: |
| Three banks, six aligned photos | **67.53%** | **67.53%** | **67.53%** | **81.82%** |
| Three banks, one photo | **66.71%** | **66.62%** | **65.26%** | **81.64%** |
| Anomalib PatchCore | 7.30% | 6.83% | 4.42% | 9.09% |
| Anomalib PaDiM | 2.20% | 2.14% | 1.78% | 4.69% |
| Anomalib STFPM | 2.09% | 2.08% | 1.95% | 4.59% |

PatchCore led the Anomalib runs. The single-photo ensemble gained 59.79 percentage points over PatchCore under changed lighting in the recorded setups. The ensemble was tuned on the public split. Input size, model tuning, and scoring-grid differences matter when interpreting the gap. [Project comparison](../README.md#13-testing-anomalib-on-can) · [Saved ensemble measurements](../results/can_ensemble_multiview.json)

---

## 1. Question

Does anomalib (PatchCore / PaDiM / STFPM) beat our champion on the Can public split of
MVTec AD 2, when scored with the project's own harness?

Reference: **67.53 all6 / 67.53 changed5** (measured twice).

Champion definition:

- detector ensemble over seeds **{42, 7, 99}**, `eps = 1.0`;
- per-capture map = mean of the champion maps of the **same scene's lighting views** ("P3");
- scored with `benchmark.summarise_by_light(threshold=0.0)`.

---

## 2. Data

Dataset root: `data/mvtec_ad_2/can` (MVTec AD 2, category `can`).

| Split | Images | Lighting |
|---|---|---|
| `train/good` | 412 | **regular only** |
| `validation/good` | 46 | regular only |
| `test_public/good` | 72 | 12 per lighting × 6 |
| `test_public/bad` | 90 | 15 per lighting × 6 |

Lightings: `regular`, `overexposed`, `underexposed`, `shift_1`, `shift_2`, `shift_3`.

Native geometry:

- images: **2232 × 1024** (W × H), RGB PNG;
- ground-truth masks: **1024 × 2232**, 2D (stored as a grayscale PNG), one per bad image;
- good and bad captures **share filenames** (e.g. `000_regular.png` exists on both sides), so
  matching predictions to captures must use the **full relative path**, not the basename.

### Defect scale (measured from the 90 bad masks)

Connected components ≥ 4 px, 108 regions total:

| Statistic | Value |
|---|---|
| median area | **100.5 px** (0.0044% of the image) |
| p10 / p90 area | 9 px / 409 px |
| max area | 446 px |
| median equivalent diameter | **11.3 px** |
| p10 / p90 diameter | 3.4 px / 22.8 px |
| components < 8 px diameter | 28% |
| components < 16 px diameter | **72%** |
| components < 32 px diameter | **100%** |

The audit uses four-way connectivity. The feature grid covers a wider area than the median defect.

---

## 3. Scoring harness

The adapter reads the continuous `anomaly_map` from Anomalib’s prediction output. Anomalib’s default post-processor stayed enabled. The adapter converts the defective masks to 2D grayscale and resizes their maps bilinearly to 1,024 × 2,232. The good-image maps enter scoring at 256 × 558, since those captures have no masks. The native ensemble contributes sixteen times as many background pixels per good capture, which affects the pooled false-positive rate.

A 4,096-bin histogram pools background pixels within each light and assigns equal weight to each eight-connected defect region. The project evaluator sweeps pixel thresholds, integrates region overlap up to 5% pixel false positives, and divides by 0.05. The default part-decision cutoff of 0.0 applies to the decision fields.

- **all6** averages AU-PRO across all six lights.
- **changed5** averages AU-PRO across overexposed, underexposed, and Shifts 1–3.
- **worst** is the lowest light-specific AU-PRO.
- **tiny** averages tiny-region coverage at 5% pixel false positives. Tiny regions occupy at most 0.01% of the native image.

The adapter matched all 162 predictions by full dataset-relative path. [Metric code](../local_contrast/evaluation.py) · [Lighting aggregation](../local_contrast/benchmark.py)

---

## 4. anomalib configuration

| Setting | Value |
|---|---|
| anomalib | **v2.7.2.dev0** (installed from source) |
| Python / stack | Python 3.11, PyTorch, Lightning |
| GPU | NVIDIA RTX 5060 Laptop, **8 GiB VRAM**, WDDM |
| System RAM | **32 GB** |
| `image_size` (H, W) | **(256, 558)** |
| effective downscale | **4× per axis (16× fewer pixels)** vs 1024×2232 |
| train batch size | 8 |
| eval batch size | 16 |
| dataloader workers | 8 (anomalib default) |
| preprocessing | `Resize((256, 558), antialias=True)` + ImageNet normalization |

Model defaults used (no algorithm tuning):

| Model | Backbone | Layers | Notes |
|---|---|---|---|
| PatchCore | `wide_resnet50_2` | `layer2`, `layer3` | coreset sampling ratio 0.1 (default) |
| PaDiM | `resnet18` | `layer1`–`layer3` | Gaussian per position using 100 retained features |
| STFPM | `resnet18` | `layer1`–`layer3` | student–teacher; checkpoint at epoch 130 |

### Why the runs resized to 558 × 256

An earlier native attempt with batch 32 ran out of GPU memory. At 558 × 256, the 32 × 70 PatchCore feature grid across 412 images gives 922,880 embeddings of 1,536 float32 features. The embedding store takes about 5.67 GB before k-center selection and 0.57 GB for the 92,287-patch coreset. The projected full-size store is about 90.4 GB. The default PaDiM backbone retains 100 features per position. Its projected native covariance alone takes about 5.71 GB for 142,848 positions, before embeddings and inversion workspace.

Anomalib accepts dimensions in height × width order. The reported runs used `(256, 558)` to keep the image’s aspect ratio. The first attempt used `(560, 256)`, which produced portrait-shaped inputs.

---

## 5. How it was run

Main script: `research_extra12.py`.

Pipeline per model:

```python
dm = MVTecAD2(root=REPO / "data" / "mvtec_ad_2",
              category="can", test_type="public",
              train_batch_size=8, eval_batch_size=16)
dm.setup()

model = Patchcore(pre_processor=Patchcore.configure_pre_processor(image_size=(256, 558)))
# ... same for PaDiM and STFPM ...

engine = Engine(default_root_dir=RUN_OUTPUT)
engine.fit(model=model, datamodule=dm)
preds = engine.predict(model=model, datamodule=dm, return_predictions=True)
```

Then:

1. Flatten anomalib's predict output (`list[Batch]` dataclasses) into per-image records.
2. Match each `image_path` to a project `Capture` by **full relative path**.
3. Read the mask as 2D: `np.asarray(read_image(cap.mask).convert("L")) != 0`.
4. For defective captures, resize the returned anomaly map bilinearly to the native mask shape (1024×2232).
5. Wrap into `ImageResult` and call `summarise_by_light(by_light)`.
6. Print `all6 / changed5 / worst / tiny`.

Commands:

```powershell
python -m anomalib_testing.research_extra12
python -m anomalib_testing.research_stfpm_eval
```

### Run timings

| Stage | Time |
|---|---|
| PatchCore fit (dominated by coreset selection) | **12,155 s ≈ 3 h 23 min** |
| PatchCore predict | 85 s |
| PaDiM fit | 58 s |
| PaDiM predict | 60 s |
| STFPM train (130 epochs, stopped) | ~2 h 06 min |
| STFPM predict (from checkpoint) | 65 s |
| Harness evaluation | ~1–2 min per model |

Prediction coverage: **162/162 test images**, 0 unmatched, all six lighting groups present.

Most of PatchCore’s fit time was spent selecting **92,287 reference patches** at about 7.8 selections per second. PaDiM fits a distribution at each position, while STFPM trains the student across multiple epochs.

---

## 6. Results

| Method | all6 | changed5 | worst light | tiny | implied regular* |
|---|---|---|---|---|---|
| Three banks, six aligned views | **67.53%** | **67.53%** | **67.53%** | **81.82%** | 67.53% |
| Three banks, one view | **66.71%** | **66.62%** | **65.26%** | **81.64%** | 67.16% |
| anomalib PatchCore | 7.30% | 6.83% | 4.42% | 9.09% | ~9.7% |
| anomalib PaDiM | 2.20% | 2.14% | 1.78% | 4.69% | ~2.5% |
| anomalib STFPM | 2.09% | 2.08% | 1.95% | 4.59% | ~2.1% |

\* `regular = 6·all6 − 5·changed5`, derived from the reported means.

---

## 7. What could explain the score gap

At 558 × 256, PatchCore’s 32 × 70 grid gives roughly one location per 32 × 32 native pixels. PaDiM and STFPM use a finest grid of 64 × 139, roughly 16 × 16 native pixels per location. The median defect is 11.3 pixels in diameter. Upsampling the maps aligns them to the masks after resizing has removed fine detail. The native ensemble uses overlapping tiles and a 25 × 25 local dark-contrast signal.

All fitting images came from regular light. PatchCore reached approximately 9.7% there and 6.83% across the changed lights. PaDiM reached approximately 2.5% and 2.14%, respectively. STFPM reached approximately 2.1% and 2.08%. The regular-light estimates use six times the reported all-light mean minus five times the reported changed-light mean.

The ensemble was tuned on the public split and averages three fitted banks. Its six-photo version also averages aligned views of the same can. The comparison reflects complete configurations and scoring grids. The effect of resizing was tested separately in the controlled [Can resolution experiment](../README.md#the-first-can-resolution-test).

---

## 8. Native-resolution follow-ups

At full size, PatchCore’s stride-eight grid would sample every eight native pixels. A tiled run could preserve more input detail while controlling the memory bank. The estimated sizes above cover individual tensors. A full-resolution STFPM pass would process sixteen times as many input pixels. All measured Anomalib scores here come from the resized runs.

Possible follow-ups include tiled PatchCore with a smaller bank, intermediate input sizes of 698 × 320 or 838 × 384, STFPM with an explicit epoch budget, and illumination augmentation or aligned-view averaging. A common native-size scoring grid for good and defective captures would keep background pixel weighting consistent.

---

## 9. Engineering notes

- With pandas 3, missing mask paths for good samples arrived as `NaN`. The adapter converts them to `None` in the data table and handles missing values in path validation.
- The adapter extracts per-image fields from Anomalib's prediction batches, matches paths including the good or defective folder, and converts defect masks to 2D before scoring.
- In the recorded environment, calling `Engine()` picked up Lightning’s 1,000-epoch default for STFPM. Training stopped after the saved epoch 130 checkpoint at global step 6,812. The checkpoint occupies about 32 MiB.
- The original PowerShell logs contain embedded NUL characters in their progress output. Decode as UTF-16 when applicable to read the recorded summary lines.

---

## 10. Reproduce and inspect the runs

Install the [Anomalib source build](https://github.com/open-edge-platform/anomalib/tree/3fd9805e1de959c13ad93a24f3f6fae9a7e920a4) (`2.7.2.dev0`) and its dependencies. Extract MVTec AD 2 under `data/mvtec_ad_2/`. Run these commands from the repository root.

```sh
python -m anomalib_testing.research_extra12
python -m anomalib_testing.research_stfpm_eval
```

The first command fits all three models. The second evaluates the recorded STFPM checkpoint independently and can run once the LFS weights are present. Run `git lfs pull` if the checkout has pointer files. Fresh runs write to `artifacts/anomalib_testing/`.

| Artifact | Contents |
| --- | --- |
| [Fit and scoring runner](research_extra12.py) | Fits the Anomalib models and scores predictions with the project evaluator. |
| [STFPM replay](research_stfpm_eval.py) | Scores the saved epoch 130 checkpoint. |
| [Fit log](logs/research_extra12.log) and [STFPM scoring log](logs/research_stfpm_eval.log) | Original timing and score records. |
| [PatchCore run](results/Patchcore/MVTecAD2/can/v5/) | Checkpoint and 162 preview images. |
| [PaDiM run](results/Padim/MVTecAD2/can/v0/) | Checkpoint and 162 preview images. |
| [STFPM run](results/Stfpm/MVTecAD2/can/v0/) | Checkpoint and 162 preview images. |

The preview images contain resized captures from **MVTec AD 2 Can** with model-generated heatmaps. The MVTec images and the derived previews are shared under **CC BY-NC-SA 4.0**. [Dataset and terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) · [Dataset paper and authors](https://link.springer.com/article/10.1007/s11263-026-02743-0) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
