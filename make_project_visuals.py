"""Render source-backed inspection figures from the recorded Can experiments."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from multiscale_experiment import add_fusion, build_reference, infer
from run_experiment import read_image


OUTPUT = Path("results/visuals")
CACHE = Path("artifacts/project_visuals")
DATASET = Path("mvtec_ad_2_dataset")
BG = "#f4f2ec"
INK = "#17262c"
MUTED = "#53626a"
GREEN = "#137466"
ORANGE = "#c05222"
BORDER = "#d3d9d6"
LIGHTS = ("regular", "underexposed", "overexposed", "shift_1", "shift_2", "shift_3")
HERO_SCENE = "003"
MISS_SCENE = "000"
FALSE_SCENE = "003"
FONT_ROOT = Path("C:/Windows/Fonts")
MANIFEST = {
    "dataset": {
        "title": "MVTec AD 2",
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
        "license": "CC BY-NC-SA 4.0",
    },
    "display": {
        "anomaly_color": "inferno palette; min=0 and max=1.25 times each recorded normal-validation threshold",
        "score": "raw image maximum divided by that variant's recorded rejection threshold; a dimensionless ratio",
        "binary": "raw fusion map >= recorded rejection threshold",
        "mask": "original capture-specific binary annotation, resized with nearest-neighbor interpolation",
    },
    "figures": {},
}


def font(size, bold=False):
    preferred = FONT_ROOT / ("segoeuib.ttf" if bold else "segoeui.ttf")
    if preferred.is_file():
        return ImageFont.truetype(str(preferred), size)

    from matplotlib import get_data_path

    fallback_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    fallback = Path(get_data_path()) / "fonts/ttf" / fallback_name
    return ImageFont.truetype(str(fallback), size)


def text(canvas, xy, value, size=24, color=INK, bold=False):
    ImageDraw.Draw(canvas).text(xy, value, font=font(size, bold), fill=color)


def figure(title, subtitle, height):
    canvas = Image.new("RGB", (1800, height), BG)
    text(canvas, (48, 26), title, 43, bold=True)
    text(canvas, (50, 89), subtitle, 23, MUTED)
    return canvas


def finish(canvas, name, provenance):
    draw = ImageDraw.Draw(canvas)
    y = canvas.height - 48
    draw.line((48, y - 12, 1752, y - 12), fill=BORDER, width=2)
    text(canvas, (50, y), "MVTec AD 2 · CC BY-NC-SA 4.0 · public research captures · colors show anomaly distance / threshold", 19, MUTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    canvas.save(path, optimize=True)
    MANIFEST["figures"][name] = {"dimensions": list(canvas.size), **provenance}
    print(f"Rendered {path}", flush=True)


def paste_fit(canvas, image, box, *, nearest=False, background="white"):
    x, y, w, h = box
    panel = Image.new("RGB", (w, h), background)
    preview = image.convert("RGB")
    ratio = min(w / preview.width, h / preview.height)
    size = (max(1, round(preview.width * ratio)), max(1, round(preview.height * ratio)))
    preview = preview.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    panel.paste(preview, ((w - preview.width) // 2, (h - preview.height) // 2))
    canvas.paste(panel, (x, y))
    ImageDraw.Draw(canvas).rectangle((x, y, x+w-1, y+h-1), outline=BORDER, width=1)


def source(label, number, light):
    return DATASET / "can/test_public" / label / f"{number}_{light}.png"


def truth(number, light):
    path = DATASET / "can/test_public/ground_truth/bad" / f"{number}_{light}_mask.png"
    return np.asarray(Image.open(path)) != 0


def crop_box(mask, margin=45, minimum=128):
    y, x = np.nonzero(mask)
    center_x, center_y = (x.min()+x.max())/2, (y.min()+y.max())/2
    side = max(minimum, int(max(x.max()-x.min()+1, y.max()-y.min()+1) + margin*2))
    left = max(0, min(mask.shape[1]-side, int(center_x-side/2)))
    top = max(0, min(mask.shape[0]-side, int(center_y-side/2)))
    return (left, top, left+side, top+side)


def heat_overlay(image, score_map, threshold, *, mask=None):
    from matplotlib import colormaps

    ratio = np.clip(score_map / max(threshold, 1e-9), 0, 1.25) / 1.25
    colors = colormaps["inferno"](ratio)[..., :3] * 255
    alpha = (0.66 * ratio)[..., None]
    pixels = np.asarray(image.convert("RGB"))
    overlay = Image.fromarray(np.uint8(np.clip(pixels*(1-alpha) + colors*alpha, 0, 255)))
    if mask is not None:
        # Mark the ground-truth boundary in cyan while retaining the real pixels.
        from scipy.ndimage import binary_erosion
        boundary = mask & ~binary_erosion(mask)
        data = np.array(overlay)
        data[boundary] = (35, 226, 218)
        overlay = Image.fromarray(data)
    return overlay


def mask_image(mask):
    return Image.fromarray(np.uint8(mask)*255).convert("RGB")


def metadata(run):
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))


def score_row(run, label, scene, light):
    target = f"can/{label}/{scene}"
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row["scene"] == target and row["lighting"] == light)


def maps_for(run_name, requests, device):
    run = Path("artifacts/multiscale") / run_name / "can"
    meta = metadata(run)
    digest = hashlib.sha256((run / "metrics.json").read_bytes()).hexdigest()[:12]
    needed = []
    collected = {}
    for label, scene, light in requests:
        cache = CACHE / f"{run_name}-{digest}-{label}-{scene}-{light}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                collected[(label, scene, light)] = {
                    key: saved[key].copy() for key in ("context", "fine", "fusion")
                }
        else:
            needed.append((label,scene,light,cache))
    if needed:
        print(f"Refitting {run_name} from {len(meta['training_images'])} recorded good source images", flush=True)
        args = SimpleNamespace(
            treatment=meta["treatment"], batch_size=meta["batch_size"],
            backbone=meta.get("backbone", "resnet"), bank_size=meta["context_bank"],
            fine_bank_size=meta["fine_bank"], seed=meta["seed"], train_limit=0,
            weights=Path("artifacts/backbones/convnext_tiny_dinov3.safetensors"),
        )
        paths = [DATASET / value for value in meta["training_images"]]
        detector, _, _ = build_reference(paths,args,device)
        CACHE.mkdir(parents=True, exist_ok=True)
        for label, scene, light, cache in needed:
            image = read_image(source(label, scene, light))
            maps = add_fusion(infer(detector, image, args), meta["fusion_scales"])
            expected = score_row(run,label,scene,light)
            gaps = {key:abs(float(maps[key].max())-float(expected[key])) for key in maps}
            if max(gaps.values()) > 0.0002:
                raise ValueError(f"Regenerated scores differ from saved run: {run_name} {scene} {light} {gaps}")
            np.savez_compressed(cache, **maps)
            collected[(label,scene,light)] = maps
            print(f"Verified {run_name}/{label}/{scene}/{light}; maximum score difference {max(gaps.values()):.2g}", flush=True)
        del detector
        if device == "cuda":
            torch.cuda.empty_cache()
    checks = {}
    for (label, scene, light), maps in collected.items():
        expected = score_row(run, label, scene, light)
        checks[f"{label}/{scene}/{light}"] = {
            key: abs(float(values.max()) - float(expected[key]))
            for key, values in maps.items()
        }
        if max(checks[f"{label}/{scene}/{light}"].values()) > 0.0002:
            raise ValueError(f"Cached maps disagree with recorded scores for {run_name}")
    MANIFEST.setdefault("regeneration_checks", {}).setdefault(run_name, {}).update(checks)
    return meta, collected


def scene_provenance(label, scene, light, roi=None):
    entry = {"image":source(label,scene,light).as_posix(), "scene":f"can/{label}/{scene}", "lighting":light}
    if label == "bad":
        entry["mask"] = f"mvtec_ad_2_dataset/can/test_public/ground_truth/bad/{scene}_{light}_mask.png"
    if roi is not None:
        entry["crop_xyxy"] = list(map(int,roi))
    return entry


def hero(meta, maps):
    image = read_image(source("bad",HERO_SCENE,"regular"))
    mask = truth(HERO_SCENE,"regular")
    roi = crop_box(mask,margin=35,minimum=112)
    mapping = maps[("bad",HERO_SCENE,"regular")]["fusion"]
    threshold = meta["score_thresholds"]["fusion"]
    ratio = float(mapping.max()/threshold)
    overlay = heat_overlay(image,mapping,threshold)
    marked = image.copy()
    ImageDraw.Draw(marked).rectangle(roi, outline=ORANGE,width=6)
    canvas = figure("Inspecting a 131-pixel defect", "Can scene 003 · regular light · first correctly rejected regular-light defective capture", 1110)
    text(canvas,(50,143),"Camera image · 2,232 × 1,024",26,bold=True)
    paste_fit(canvas,marked,(50,187,820,376))
    text(canvas,(920,143),"Native detail",26,bold=True)
    paste_fit(canvas,image.crop(roi),(920,187,350,350))
    text(canvas,(1320,143),"Annotated defect",26,bold=True)
    paste_fit(canvas,mask_image(mask).crop(roi),(1320,187,350,350),nearest=True,background="black")
    text(canvas,(50,609),"Frozen native-scale fusion heatmap",26,bold=True)
    paste_fit(canvas,overlay,(50,653,820,376))
    text(canvas,(920,609),"Heatmap at the same crop",26,bold=True)
    paste_fit(canvas,heat_overlay(image,mapping,threshold,mask=mask).crop(roi),(920,653,350,350))
    text(canvas,(1320,609),"Thresholded prediction",26,bold=True)
    paste_fit(canvas,mask_image(mapping>=threshold).crop(roi),(1320,653,350,350),nearest=True,background="black")
    text(canvas,(920,548),"112 × 112 source pixels",22,MUTED)
    text(canvas,(920,1014),f"REJECT · score / threshold {ratio:.3f} · cyan = true boundary",24,GREEN,bold=True)
    finish(canvas,"inspection_hero.png",{
        **scene_provenance("bad",HERO_SCENE,"regular",roi),
        "selection":"Lowest numeric regular-light defective scene whose recorded native-fusion score exceeds its source-validation threshold.",
        "run":"artifacts/multiscale/native/can", "threshold":threshold,"score":float(mapping.max()),
        "mask_pixels": int(mask.sum()),
        "predicted_pixels": int(np.count_nonzero(mapping >= threshold)),
        "correctly_predicted_pixels": int(np.count_nonzero(mask & (mapping >= threshold))),
        "illustrative": True,
    })


def branches(meta,maps):
    image=read_image(source("bad",HERO_SCENE,"regular"))
    mask=truth(HERO_SCENE,"regular")
    roi=crop_box(mask,margin=65,minimum=160)
    canvas=figure("Where the model looks", "Same capture and crop · each heatmap uses its own recorded normal-validation threshold",730)
    items=[("Original pixels",image.crop(roi),None), *[
        ({"context":"Context · stride 8","fine":"Detail · stride 4","fusion":"Equal-weight fusion"}[key],
         heat_overlay(image,maps[("bad",HERO_SCENE,"regular")][key],meta["score_thresholds"][key],mask=mask).crop(roi),key)
        for key in ("context","fine","fusion")
    ]]
    for index,(label,preview,key) in enumerate(items):
        x=50+index*435
        text(canvas,(x,155),label,26,bold=True)
        paste_fit(canvas,preview,(x,205,395,395))
        if key:
            ratio=float(maps[("bad",HERO_SCENE,"regular")][key].max()/meta["score_thresholds"][key])
            text(canvas,(x,613),f"score / threshold {ratio:.3f}",22,MUTED)
        else:
            text(canvas,(x,613),"Cyan on overlays = truth",22,MUTED)
    finish(canvas,"feature_branches.png",{**scene_provenance("bad",HERO_SCENE,"regular",roi),"run":"artifacts/multiscale/native/can","map_variants":["context","fine","fusion"]})


def resolution(native_meta,native,full_meta,full):
    image=read_image(source("bad",HERO_SCENE,"regular"))
    mask=truth(HERO_SCENE,"regular")
    roi=crop_box(mask,margin=65,minimum=160)
    canvas=figure("The cost of shrinking the whole image", "Same source image, 96 normal fitting images, and bank budgets · source-calibrated fusion in both modes",730)
    views=[
        ("Original detail",image.crop(roi),"160 × 160 source pixels"),
        ("Full frame → 256 input",heat_overlay(image,full[("bad",HERO_SCENE,"regular")]["fusion"],full_meta["score_thresholds"]["fusion"],mask=mask).crop(roi),"Upsampled full-frame response"),
        ("Native 256-pixel tiles",heat_overlay(image,native[("bad",HERO_SCENE,"regular")]["fusion"],native_meta["score_thresholds"]["fusion"],mask=mask).crop(roi),"Overlapping high-detail response"),
        ("Ground-truth annotation",mask_image(mask).crop(roi),"131 annotated pixels"),
    ]
    for index,(label,view,note) in enumerate(views):
        x=50+index*435
        text(canvas,(x,160),label,25,bold=True)
        paste_fit(canvas,view,(x,210,395,395),nearest=index==3)
        text(canvas,(x,620),note,21,MUTED)
    finish(canvas,"resolution_comparison.png",{**scene_provenance("bad",HERO_SCENE,"regular",roi),"runs":["artifacts/multiscale/full/can","artifacts/multiscale/native/can"],"display_normalization":"Each score map divided by that model's own source-validation threshold; all color palettes span ratio 0–1.25."})


def lighting(meta,maps):
    roi=crop_box(truth(HERO_SCENE,"regular"),margin=65,minimum=160)
    canvas=figure("One part, six real lighting conditions", "Can scene 003 · same native-scale model and fixed regular-light threshold · left pixels, right heatmap",1175)
    threshold=meta["score_thresholds"]["fusion"]
    records=[]
    for index,light in enumerate(LIGHTS):
        x=50+(index%3)*575
        y=165+(index//3)*440
        image=read_image(source("bad",HERO_SCENE,light))
        mask=truth(HERO_SCENE,light)
        values=maps[("bad",HERO_SCENE,light)]["fusion"]
        score = float(values.max())
        rejected = score >= threshold
        text(canvas,(x,y),light.replace("_"," ").title(),27,bold=True)
        paste_fit(canvas,image.crop(roi),(x,y+48,250,290))
        paste_fit(canvas,heat_overlay(image,values,threshold,mask=mask).crop(roi),(x+260,y+48,250,290))
        text(canvas,(x,y+352),f"{'REJECT' if rejected else 'PASS'} · score / threshold {score/threshold:.3f}",24,GREEN if rejected else ORANGE,bold=True)
        records.append({**scene_provenance("bad",HERO_SCENE,light,roi),"score":score,"threshold":threshold,"rejected":rejected})
    text(canvas,(50,1058),"The crop is fixed across all six captures. Each cyan boundary comes from that lighting condition's own annotation.",21,MUTED)
    finish(canvas,"lighting_comparison.png",{"run":"artifacts/multiscale/native/can","captures":records})


def failure(meta,maps):
    threshold=meta["score_thresholds"]["fusion"]
    cases=[("bad",MISS_SCENE,"regular","Defective part passed"),("good",FALSE_SCENE,"shift_1","Good part rejected")]
    canvas=figure("Two decisions the model gets wrong", "Recorded native-fusion threshold · a missed tiny defect and a false alarm on a verified-good part",900)
    provenance=[]
    for index,(label,scene,light,title) in enumerate(cases):
        x=50+index*875
        image=read_image(source(label,scene,light))
        values=maps[(label,scene,light)]["fusion"]
        mask=truth(scene,light) if label=="bad" else np.zeros_like(values,dtype=bool)
        if label=="bad":
            roi=crop_box(mask,margin=65,minimum=160)
        else:
            ypeak,xpeak=np.unravel_index(np.argmax(values),values.shape)
            blank = np.zeros_like(mask)
            blank[ypeak, xpeak] = True
            roi=crop_box(blank,margin=80,minimum=160)
        text(canvas,(x,150),title,30,ORANGE,bold=True)
        text(canvas,(x,197),f"Can {label}/{scene} · {light}",22,MUTED)
        marked=image.copy()
        ImageDraw.Draw(marked).rectangle(roi,outline=ORANGE,width=6)
        paste_fit(canvas,marked,(x,239,800,245))
        paste_fit(canvas,image.crop(roi),(x,515,240,240))
        paste_fit(canvas,heat_overlay(image,values,threshold,mask=mask if label=="bad" else None).crop(roi),(x+255,515,240,240))
        final=mask_image(mask if label=="bad" else np.zeros_like(mask)).crop(roi)
        paste_fit(canvas,final,(x+510,515,240,240),nearest=True)
        text(canvas,(x,772),f"score / threshold {float(values.max()/threshold):.3f} · {'PASS' if values.max()<threshold else 'REJECT'}",24,ORANGE,bold=True)
        provenance.append({**scene_provenance(label,scene,light,roi),"score":float(values.max()),"threshold":threshold,"selection":"Miss = lowest numeric annotated scene. False reject = lowest numeric good scene falsely rejected by shift_1."})
    finish(canvas,"failure_case.png",{"run":"artifacts/multiscale/native/can","captures":provenance})


def trial_gallery(device, use_saved=False):
    trials=[
        ("full","Full-frame control"),("native","Native RGB"),("luma","Bounded gamma"),
        ("retinex","Mild Retinex"),("augmented","Training-light simulation"),("coreset","Farthest-first reference"),
        ("convnext-native","DINOv3 ConvNeXt-Tiny"),("gradient-native","Sobel + Laplacian detail"),
        ("synthetic-native","Synthetic-defect head"),("distillation-native","Student–teacher detail"),
    ]
    for part in range(2):
        canvas=figure("Can experiment gallery",f"Part {part+1}/2 · the same first defective scene, bad/000, under regular lighting · fusion maps from recorded experiments",1270)
        capture_records=[]
        for index,(name,label) in enumerate(trials[part*5:part*5+5]):
            run=Path("artifacts/multiscale")/name/"can"
            meta=metadata(run)
            saved=run/"regular-fusion.png"
            roi=crop_box(truth("000","regular"),margin=50,minimum=128)
            image=read_image(source("bad","000","regular"))
            if use_saved:
                with Image.open(saved) as previous:
                    panel_width=(previous.width-48)//3
                    preview=previous.crop((panel_width+24,62,2*panel_width+24,previous.height-12)).convert("RGB")
                native_roi=None
            else:
                _, regenerated=maps_for(name,[("bad","000","regular")],device)
                values=regenerated[("bad","000","regular")]["fusion"]
                preview=heat_overlay(image,values,meta["score_thresholds"]["fusion"])
                native_roi=heat_overlay(image,values,meta["score_thresholds"]["fusion"],mask=truth("000","regular")).crop(roi)
            x=50+(index%2)*875
            y=155+(index//2)*330
            text(canvas,(x,y),label,28,bold=True)
            if native_roi is None:
                paste_fit(canvas,preview,(x,y+50,795,225))
            else:
                paste_fit(canvas,preview,(x,y+50,535,225))
                paste_fit(canvas,native_roi,(x+550,y+50,225,225))
            row=score_row(run,"bad","000","regular")
            ratio=float(row["fusion"])/meta["score_thresholds"]["fusion"]
            verdict="REJECT" if row["fusion_rejected"]=="True" else "PASS"
            text(canvas,(x,y+287),f"{verdict} · score / threshold {ratio:.3f}",22,GREEN if verdict=="REJECT" else ORANGE,bold=True)
            capture_records.append({
                "run": run.as_posix(),
                "recorded_predictions": (run / "predictions.csv").as_posix(),
                "recorded_metrics": (run / "metrics.json").as_posix(),
                **scene_provenance("bad", "000", "regular", roi),
                "map_source": "saved thumbnail" if use_saved else "regenerated native-coordinate fusion",
                "fusion_score": float(row["fusion"]),
                "threshold": meta["score_thresholds"]["fusion"],
            })
        x,y=925,815
        image=read_image(source("bad","000","regular"))
        roi=crop_box(truth("000","regular"),margin=50,minimum=128)
        text(canvas,(x,y),"Source detail and true mask",28,bold=True)
        paste_fit(canvas,image.crop(roi),(x,y+50,225,225))
        paste_fit(canvas,mask_image(truth("000","regular")).crop(roi),(x+245,y+50,225,225),nearest=True)
        text(canvas,(x+495,y+57),"60 annotated pixels",23,MUTED)
        text(canvas,(x+495,y+96),"Native crops",23,MUTED)
        text(canvas,(x+495,y+135),"for visibility",23,MUTED)
        text(canvas,(50,1160),"Each full map and native ROI uses its own fixed validation threshold. Cyan marks truth. This is one difficult capture.",20,MUTED)
        finish(canvas,f"can_trials_0{part+1}.png",{"captures":capture_records,"source_detail":scene_provenance("bad","000","regular",roi),"selection":"The first defective regular-light capture, held constant for every configuration."})



def category_examples():
    categories = (
        ("can", "Can"), ("fabric", "Fabric"), ("fruit_jelly", "Fruit jelly"),
        ("rice", "Rice"), ("sheet_metal", "Sheet metal"), ("vial", "Vial"),
        ("wallplugs", "Wall plugs"), ("walnuts", "Walnuts"),
    )
    canvas = figure(
        "Eight materials, one inspection pipeline",
        "First defective regular-light capture per category · recorded native-fusion input and heatmap previews",
        1555,
    )
    captures = []
    for index, (category, label) in enumerate(categories):
        run = Path("artifacts/multiscale/native") / category
        meta = metadata(run)
        with (run / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            row = next(
                row for row in csv.DictReader(stream)
                if row["is_defective"] == "True" and row["lighting"] == "regular"
            )
        saved = run / "regular-fusion.png"
        with Image.open(saved) as panels:
            width = (panels.width - 48) // 3
            bottom = panels.height - 12
            raw = panels.crop((12, 62, 12 + width, bottom)).convert("RGB")
            heatmap = panels.crop((24 + width, 62, 24 + 2 * width, bottom)).convert("RGB")
        x = 50 + (index % 2) * 875
        y = 160 + (index // 2) * 315
        verdict = "REJECT" if row["fusion_rejected"] == "True" else "PASS"
        ratio = float(row["fusion"]) / meta["score_thresholds"]["fusion"]
        text(canvas, (x, y), label, 28, bold=True)
        text(canvas, (x + 385, y + 3), f"{verdict} · ratio {ratio:.3f}", 23,
             GREEN if verdict == "REJECT" else ORANGE, bold=True)
        text(canvas, (x, y + 41), "Input", 20, MUTED)
        text(canvas, (x + 400, y + 41), "Fusion heatmap", 20, MUTED)
        paste_fit(canvas, raw, (x, y + 73, 385, 205))
        paste_fit(canvas, heatmap, (x + 400, y + 73, 385, 205))
        captures.append({
            "run": run.as_posix(), "saved_overlay": saved.as_posix(),
            "image": (DATASET / row["image"]).as_posix(), "scene": row["scene"],
            "mask": (DATASET / category / f"test_public/ground_truth/bad/{Path(row['image']).stem}_mask.png").as_posix(),
            "lighting": "regular", "fusion_score": float(row["fusion"]),
            "threshold": meta["score_thresholds"]["fusion"],
            "map_source": "input and heatmap panels cropped from the saved thumbnail preview",
            "saved_input_box": [12, 62, 12 + width, bottom],
            "saved_heatmap_box": [24 + width, 62, 24 + 2 * width, bottom],
        })
    text(canvas, (50, 1430), "All examples are defective. PASS marks a missed defective capture. Compare category-level results alongside these examples.", 21, MUTED)
    finish(canvas, "category_examples.png", {
        "captures": captures,
        "selection": "First defective regular-light capture in each category, using the exact saved native-fusion preview.",
        "display": "Saved evaluation overlays normalize scores by their own validation threshold, clipping at 1.0 with a maximum color opacity of 0.7.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gallery-only", action="store_true")
    args = parser.parse_args()

    optional_deps = Path("artifacts/superadd-deps")
    if optional_deps.is_dir():
        sys.path.insert(0, str(optional_deps.resolve()))

    category_examples()
    trial_gallery(args.device, use_saved=args.gallery_only)
    if not args.gallery_only:
        requests = [("bad", HERO_SCENE, light) for light in LIGHTS]
        requests.extend([
            ("bad", MISS_SCENE, "regular"),
            ("good", FALSE_SCENE, "shift_1"),
        ])
        meta, native = maps_for("native", requests, args.device)
        full_meta, full = maps_for("full", [("bad", HERO_SCENE, "regular")], args.device)
        hero(meta, native)
        branches(meta, native)
        resolution(meta, native, full_meta, full)
        lighting(meta, native)
        failure(meta, native)

    (OUTPUT / "manifest.json").write_text(
        json.dumps(MANIFEST, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__=="__main__":
    main()
