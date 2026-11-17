"""Score incoming parts or build a local private-split benchmark archive."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import ImageResult, save_overlay
from export_benchmark import BenchmarkExport, MASK_DIR, SCORE_DIR
from multiscale_experiment import (CATEGORIES, VARIANTS, add_fusion, add_model_args, build_reference,
                                   calibrate, configure, infer)
from part_scoring import local_contrast_score
from run_experiment import read_image


def fit_category(category, args, device):
    root = args.dataset / category
    training = sorted((root / "train" / "good").glob("*_regular.png"))
    validation = sorted((root / "validation" / "good").glob("*_regular.png"))
    if not training or not validation:
        raise ValueError(f"Missing regular-light normal train or validation images under {root}")
    detector, selected, seconds = build_reference(training, args, device)
    scales, thresholds, scores = calibrate(detector, validation, args)
    print(f"{category} reference | {len(selected)} train | {len(scores)} validation | {seconds:.1f}s fit", flush=True)
    return detector, scales, thresholds


def selected_variant(method, maps, scales):
    name = method.removesuffix("_local")
    if name == "fusion":
        add_fusion(maps, scales)
    anomaly_map = maps[name]
    score = local_contrast_score(anomaly_map) if method.endswith("_local") else float(anomaly_map.max())
    return name, anomaly_map, score


def score_parts(args, device):
    detector, scales, thresholds = fit_category(args.category, args, device)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    verdicts = []
    for index, path in enumerate(args.images):
        maps = infer(detector, read_image(path), args)
        name, anomaly_map, score = selected_variant(args.method, maps, scales)
        threshold = thresholds[args.method]
        context_rejected = float(maps["context"].max()) >= thresholds["context"]
        fine_rejected = float(maps["fine"].max()) >= thresholds["fine"]
        decision = "reject" if score >= threshold else "pass"
        if args.review_disagreement and context_rejected != fine_rejected:
            decision = "review"
        record = {"image": str(path), "score": score, "threshold": threshold, "decision": decision}
        if args.output:
            result = ImageResult(str(path), path.stem, "incoming", False, score, anomaly_map, None)
            preview = args.output / f"{index:03d}-{path.stem}.png"
            save_overlay(path, result, preview, (0.0, thresholds[name]), decision=decision, show_mask=False)
            record["preview"] = str(preview)
        verdicts.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    if args.output:
        (args.output / "decisions.json").write_text(json.dumps(verdicts, indent=2) + "\n", encoding="utf-8")


def write_run_identity(args, output):
    # The sidecar stays outside the organizer's submission directory and
    # prevents resumed runs from silently mixing maps from different settings.
    sidecar = output.with_name(f"{output.name}_run.json")
    identity = {
        "dataset": str(args.dataset.resolve()), "method": args.method, "backbone": args.backbone,
        "treatment": args.treatment, "train_limit": args.train_limit,
        "validation_limit": args.validation_limit, "seed": args.seed,
        "bank_size": args.bank_size, "fine_bank_size": args.fine_bank_size,
        "batch_size": args.batch_size, "device": args.device,
        "max_side": args.max_side, "with_masks": args.with_masks,
    }
    if args.backbone == "convnext":
        with args.weights.open("rb") as stream:
            identity["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if sidecar.exists():
        if args.resume and json.loads(sidecar.read_text(encoding="utf-8")) == identity:
            return
        raise ValueError(f"Choose a fresh output or resume with matching settings: {sidecar}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output folder: {output}")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2)
        stream.write("\n")


def build_submission(args, device):
    exporter = BenchmarkExport(args.dataset, args.output, with_masks=args.with_masks, max_side=args.max_side)
    output = exporter.output_root
    args.device = device
    write_run_identity(args, output)
    for category in dict.fromkeys(args.categories or CATEGORIES):
        entries = list(exporter.entries(category))

        def complete(entry):
            if not (output / SCORE_DIR / entry.relative.with_suffix(".tiff")).is_file():
                return False
            return not args.with_masks or (output / MASK_DIR / entry.relative).is_file()

        if args.resume:
            entries = [entry for entry in entries if not complete(entry)]
        if not entries:
            continue
        detector, scales, thresholds = fit_category(category, args, device)
        with ThreadPoolExecutor(max_workers=2) as writer:
            pending = deque()
            for index, entry in enumerate(entries, start=1):
                maps = infer(detector, read_image(entry.source), args)
                name, anomaly_map, _ = selected_variant(args.method, maps, scales)
                mask = anomaly_map >= thresholds[name] if args.with_masks else None
                pending.append(writer.submit(exporter.write, entry, anomaly_map, binary_mask=mask))
                if len(pending) == 2:
                    pending.popleft().result()
                if index % 50 == 0:
                    print(f"{category} | inferred {index}/{len(entries)} captures", flush=True)
            for task in pending:
                task.result()
        print(f"{category} | wrote {len(entries)} captures", flush=True)
    if args.archive:
        archive = exporter.package(args.archive, checker=args.checker)
        print(f"Validated private splits | archive {archive}", flush=True)
    else:
        print(f"Maps saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Fit once and score a batch of parts")
    score.add_argument("category", choices=CATEGORIES)
    score.add_argument("images", type=Path, nargs="+")
    score.add_argument("--output", type=Path)
    score.add_argument("--review-disagreement", action="store_true")
    score.set_defaults(handler=score_parts)

    benchmark = commands.add_parser("benchmark", help="Generate the private scores without reading test labels")
    benchmark.add_argument("--categories", choices=CATEGORIES, nargs="+", help="Omit to process all eight categories")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--archive", type=Path, help="Validate all eight categories and write a tar.gz archive")
    benchmark.add_argument("--checker", type=Path, help="Optional bundled MVTec checker script for archive validation")
    benchmark.add_argument("--max-side", type=int, help="Optional aligned score-map storage resolution")
    benchmark.add_argument("--with-masks", action="store_true", help="Also export source-calibrated binary masks")
    benchmark.add_argument("--resume", action="store_true")
    benchmark.set_defaults(handler=build_submission)

    for command in (score, benchmark):
        add_model_args(command)
        command.add_argument("--method", choices=VARIANTS, default="fusion")
    args = parser.parse_args()
    if args.command == "benchmark" and args.checker and not args.archive:
        parser.error("--checker requires --archive")
    device = configure(parser, args)
    args.handler(args, device)


if __name__ == "__main__":
    main()
