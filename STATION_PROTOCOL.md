# Matched captures at a physical inspection station

Use this protocol to find out whether the camera records a specified scratch under different lighting and whether a frozen detector still catches it. The repository's completed optical replay used public MVTec AD 2 images. The steps below require independent station captures and measurements. [Public replay](results/acquisition_and_adaptation.md)

## Set up the station

1. Specify the smallest unacceptable defect width and the surface area that must be inspected. Record camera and lens models, working distance, focus setting, and the measured field of view at the part plane. Check sharpness and contrast on an optical target and representative defects across the field. Dividing field width by sensor pixels gives the sampling pitch on the part. Verify useful resolution with the target and real defects at the intended aperture and line speed. [Optical resolution](https://www.edmundoptics.com/knowledge-center/application-notes/imaging/resolution/)
2. Fix the camera and part pose. Shield the setup from ambient changes and synchronize the shutter and light if parts move. Set exposure, gain, focus, and any white-balance or tone mapping controls manually. Keep them locked for the geometry comparison where feasible. When two light recipes require different exposure, record the difference and interpret the comparison as a change in the combined recipe. Save original camera files separately and document the fixed conversion to analysis PNGs. [MVTec acquisition setups](https://link.springer.com/article/10.1007/s11263-026-02743-0)
3. Photograph each test part in the same pose under `nominal`, `diffuse`, `grazing`, and `polarized` light. Record the nominal production recipe, the diffuse fixture, the grazing angle and direction, and the polarizer and analyzer orientations. Vary scratch orientation across parts and balance light order to reduce effects of time or warm-up. On reflective parts, look for both suppressed glare and lost defect contrast when changing polarization.

Assign every physical part a stable `part_id` before imaging. Verify defects independently from the scored views, such as with a higher-magnification reference or a measured defect standard. Assign disjoint parts to `train`, `calibration`, and `test`. Use only verified-good nominal captures for training and threshold setting. Populate test with verified-good and defective parts from independent lots where possible. All views of a test part stay together. Mark defects at native resolution using the verified reference. A mask can carry across light conditions for the same part when registration has been checked. Keep defect masks outside detector fitting.

## Record and validate the captures

Keep the study under an ignored local directory such as `artifacts/station/trial-01/`. It contains `station.json`, `captures.csv`, the analysis PNGs, and native-coordinate mask PNGs. Set `data_origin` to `physical_station` for actual station data and `synthetic_test` for software checks. The JSON fields below are a template. Replace all placeholders with the observed station settings.

```json
{
  "study_id": "replace with the station study ID",
  "data_origin": "physical_station",
  "part_family": "replace with the part family",
  "camera_model": "replace with the camera model and mode",
  "lens_model": "replace with the lens model, focus and aperture",
  "fixture_id": "replace with the fixture ID",
  "image_pipeline": "describe the fixed conversion from camera output to native-size 8-bit PNG",
  "working_distance_mm": null,
  "field_width_mm": null,
  "field_height_mm": null,
  "minimum_defect_width_mm": null,
  "lighting": {
    "nominal": {"description": "record the production setup", "exposure_us": null, "gain_db": null},
    "diffuse": {"description": "record the diffuse fixture", "exposure_us": null, "gain_db": null},
    "grazing": {"description": "record the grazing angle and direction", "exposure_us": null, "gain_db": null},
    "polarized": {"description": "record both polarizer orientations", "exposure_us": null, "gain_db": null}
  }
}
```

Use one row per capture with this header. Give each good part an empty `mask` field. Use `good` or `defective` as the label and paths relative to the study folder. The checker requires equivalent image dimensions and color mode, distinct capture paths, binary native masks, complete lighting pairs, and recorded exposure and gain that match the locked recipe. A `part_id` can appear only in one split. Its physical identity and the accuracy of the external labels still depend on the capture log.

```csv
part_id,split,condition,label,image,mask,exposure_us,gain_db
```

```sh
python station_study.py validate artifacts/station/trial-01
python station_study.py analyze artifacts/station/trial-01 --output artifacts/station/trial-01-analysis.json
python station_study.py analyze artifacts/station/trial-01 --detector gradient --output artifacts/station/trial-01-detector.json
```

The acquisition-only run reports per-condition image statistics, native and resized defect contrast, and paired changes against each test part's nominal capture. With `--detector gradient` or `--detector resnet`, the runner fits the existing native-scale detector on nominal training parts and freezes its fusion threshold from the nominal calibration parts. It reports false rejects, misses, image AUROC, and native-map AU-PRO by condition. The ResNet context branch uses torchvision weights and may download them on first use. Report the number of independent parts, since four views of one part do not provide four independent observations. Measure trigger-to-decision time and any reacquisition or review workload at the station.
