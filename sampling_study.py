"""Matched native, grid-offset, and position-aware localization controls on public images."""
import argparse
import json
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation import ImageResult, summarize
from lighting_context import LightingContext
from hotspot_study import annotate, atomic_json, hotspots, native_options
from multiscale_detector import MultiScaleDetector
from multiscale_experiment import CATEGORIES, add_fusion, build_reference, infer, native_tiles, pixel_operating_point
from report_hotspots import counts, ranking, scene_interval, shortlist
from run_experiment import INPUT_SIZE, load_splits, normalize, read_image, tile_origins, training_batches

VARIANTS = ("native", "offset_mean", "offset_max", "position", "light_context")


def tile_geometry(image):
    return ((x, y, image.width, image.height) for y in tile_origins(image.height, INPUT_SIZE)
            for x in tile_origins(image.width, INPUT_SIZE))


class PositionDetector(MultiScaleDetector):
    """Append two spatial coordinates before sampling and matching reference features."""
    def __init__(self, device, scales, **reference_options):
        super().__init__(device=device, **reference_options)
        self.scales = scales
        self.geometry = iter(())

    def _feature_grids(self, batch):
        fine, context = super()._feature_grids(batch)
        geometry = [next(self.geometry) for _ in range(len(batch))]
        xywh = torch.as_tensor(geometry, device=self.device, dtype=fine.dtype)
        crop_size = xywh[:, 2:].clamp(max=INPUT_SIZE)
        resized = (crop_size * (INPUT_SIZE/crop_size.amax(dim=1, keepdim=True))).round().clamp_min(1)
        padding = ((INPUT_SIZE-resized)/2).floor()
        # fit_frame letterboxes a native tile when a source dimension is small.
        # Map padded grid cells back to the nearest real pixel location.
        def image_coordinates(centers, axis):
            fraction = ((centers-padding[:, axis, None])/resized[:, axis, None]).clamp(0,1)
            return (xywh[:, axis, None] + fraction*crop_size[:, axis, None])/xywh[:, axis+2, None]

        def append(grid, scale):
            _, h, w, _ = grid.shape
            unit_x = (torch.arange(w, device=self.device)+.5) * (INPUT_SIZE/w)
            unit_y = (torch.arange(h, device=self.device)+.5) * (INPUT_SIZE/h)
            x = image_coordinates(unit_x, 0)
            y = image_coordinates(unit_y, 1)
            coordinates = torch.stack(torch.broadcast_tensors(x[:, None, :], y[:, :, None]), -1)
            return torch.cat((grid, scale*coordinates), dim=-1)
        return append(fine, self.scales["fine"]), append(context, self.scales["context"])

    def batches(self, paths, batch_size):
        pixels, geometry = [], []
        for p in paths:
            image = read_image(p)
            for tile, location in zip(native_tiles(image), tile_geometry(image), strict=True):
                pixels.append(tile[0])
                geometry.append(location)
                if len(pixels) == batch_size:
                    self.geometry = iter(geometry)
                    yield normalize(pixels)
                    pixels, geometry = [], []
        if pixels:
            self.geometry = iter(geometry)
            yield normalize(pixels)


def shifted_copy(image):
    # Preserve the tile layout exactly. Translate the source two pixels down/right.
    pixels = np.asarray(image)
    padded = np.pad(pixels, ((2, 0), (2, 0), (0, 0)), mode="reflect")
    return Image.fromarray(padded[:image.height, :image.width])


def variant_maps(image, plain, positioned, lighting_model, scales, args):
    def fused(model, img):
        return add_fusion(infer(model, img, args), scales)["fusion"]
    start = time.perf_counter()
    native = fused(plain, image)
    base_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    positioned.geometry = tile_geometry(image)
    position = fused(positioned, image)
    pos_ms = (time.perf_counter()-start)*1000
    if lighting_model is None:
        return dict(native=native, position=position), dict(native=base_ms, position=pos_ms)
    start = time.perf_counter()
    other = fused(plain, shifted_copy(image))
    aligned = native.copy()
    aligned[:-2, :-2] = other[2:, 2:]
    shift_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    mean = (native + aligned)*.5
    mean_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    maximum = np.maximum(native, aligned)
    max_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    light_context = fused(lighting_model, image)
    light_ms = (time.perf_counter()-start)*1000
    return (dict(light_context=light_context, native=native, offset_mean=mean, offset_max=maximum, position=position),
            dict(native=base_ms, offset_mean=base_ms+shift_ms+mean_ms,
                 offset_max=base_ms+shift_ms+max_ms, position=pos_ms, light_context=light_ms))


def run(category, dataset, region_dir, output, device, position_only=False):
    variants = ("native", "position") if position_only else VARIANTS
    data = json.loads((region_dir/f"{category}.json").read_text(encoding="utf-8"))
    reference = {r["image"]: r["raw"] for r in data["validation"]+data["captures"]}
    args = native_options()
    train, validation, captures = load_splits(dataset, category, args.seed)
    native, selected, _ = build_reference(train, args, device)
    assert [p.relative_to(dataset).as_posix() for p in selected] == data["training"]
    scales = data["scales"]
    position = PositionDetector(device, scales)
    position.fit(position.batches(selected, args.batch_size))

    lighting_model = None
    if not position_only:
        lighting_model = LightingContext(device)
        lighting_model.learn_weights(selected, args.seed)
        lighting_model.fit(training_batches((read_image(p) for p in selected), INPUT_SIZE, args.batch_size))

    def score(path):
        maps, timings = variant_maps(read_image(path), native, position, lighting_model, scales, args)
        score = float(maps["native"].max())
        if abs(score-reference[path.relative_to(dataset).as_posix()]) > 0.0002:
            raise ValueError(f"Native control did not reproduce for {path}")
        return maps, timings

    calibration = {k: [] for k in variants}
    for p in validation:
        maps, _ = score(p)
        for k in variants:
            calibration[k].append(float(maps[k].max()))
    thresholds = {k: float(np.quantile(v, .90)) for k, v in calibration.items()}
    by_light, records = {}, []
    for light, items in groupby(sorted(captures, key=lambda c: c.lighting), key=lambda c: c.lighting):
        groups = {k: [] for k in variants}
        for c in items:
            maps, timings = score(c.image)
            mask = None
            if c.mask:
                with Image.open(c.mask) as stream:
                    mask = np.asarray(stream) != 0
            row = dict(image=c.image.relative_to(dataset).as_posix(), scene=c.scene,
                       lighting=light, bad=c.mask is not None, timing=timings, proposals={})
            for k, m in maps.items():
                regions = hotspots(m)
                row[k] = regions[0]["features"][0]
                if c.mask:
                    annotate(regions, mask)
                row["proposals"][k] = regions
                groups[k].append(ImageResult(row["image"], c.scene, light, row["bad"], row[k], m, mask))
            records.append(row)
        by_light[light] = {}
        for k, group in groups.items():
            metrics = summarize(group, thresholds[k])
            metrics["pixel_f1"] = pixel_operating_point(group, thresholds[k])["pixel_f1"]
            by_light[light][k] = {name: metrics[name] for name in
                                  ("au_pro_05", "tiny_pro_at_05", "image_auroc", "pixel_f1")}
        print(f"{category} {light} complete", flush=True)
    trial = "position_transfer" if position_only else "sampling"
    atomic_json(output/f"{category}_{trial}_captures.json", records)
    shifted = [r for r in records if r["lighting"] != "regular"]
    bad = [r for r in shifted if r["bad"]]
    missed = [r for r in bad if r["native"] < thresholds["native"]]
    budget = counts(shifted, "native", thresholds["native"])["false_rejects"]
    summary = {}
    for k in variants:
        def picks(rows):
            return shortlist([dict(candidates=r["proposals"][k]) for r in rows])
        summary[k] = dict(shortlist=picks(bad), baseline_misses=picks(missed),
                          decisions=counts(shifted, k, thresholds[k]),
                          delta_caught_ci=scene_interval(shifted, thresholds, True, k, "native"),
                          delta_false_rejects_ci=scene_interval(shifted, thresholds, False, k, "native"),
                          ranking=ranking(shifted, k, budget),
                          compute_ms=np.percentile([r["timing"][k] for r in shifted],[50,95]).tolist())
        for metric in ("au_pro_05", "tiny_pro_at_05", "image_auroc", "pixel_f1"):
            values = [v[k][metric] for light, v in by_light.items()
                      if light != "regular" and v[k][metric] is not None]
            summary[k][metric] = float(np.mean(values)) if values else None
    return dict(category=category, thresholds=thresholds, summary=summary, by_light=by_light,
                lighting_weights=lighting_model.weights.cpu().tolist() if lighting_model else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", choices=CATEGORIES)
    parser.add_argument("--dataset", type=Path, default=Path("mvtec_ad_2_dataset"))
    parser.add_argument("--input", type=Path, default=Path("artifacts/hotspots"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--position-only", action="store_true", help="Run only position-aware and native matching")
    opts = parser.parse_args()
    if opts.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(opts.category, opts.dataset, opts.input, opts.input, opts.device, opts.position_only)
    trial = "position_transfer" if opts.position_only else "sampling"
    atomic_json(opts.output/f"{opts.category}_{trial}.json", result, indent=2)
