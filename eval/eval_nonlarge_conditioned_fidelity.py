#!/usr/bin/env python3
"""Non-large conditioned-object fidelity for controllable generation.

Goal
----
Evaluate whether generated images faithfully preserve the **small / medium**
structures that the *condition* actually specifies. We do NOT evaluate every
YOLO object, because some YOLO detections are not represented in the condition
map. Instead we evaluate the intersection:

    evaluable object  =  (non-large YOLO box)  AND  (matched condition signal)

Size buckets (by box_area_ratio = box_area / (H*W)):
    small    : ratio <  --small_thr            (default 0.02)
    medium   : --small_thr <= ratio < --medium_thr   (0.02 .. 0.10)
    non-large: ratio < --medium_thr            (small + medium; large excluded)

Condition-specific "is this box actually conditioned?" rule
-----------------------------------------------------------
* seg  : extract connected components from the seg condition map. Keep box B if
         for some component M: IoA_cond=|B∩M|/|M| > --ioa_cond_thr
         OR IoA_box=|B∩M|/|B| > --ioa_box_thr.
* edge : count condition edge pixels inside B. Keep if count > --edge_min_pixels
         OR (edge_pixels / box_area) > --edge_box_ratio_thr.
* depth: the box itself is the eval region. Optional filter: skip boxes whose
         mean condition-depth gradient < --depth_grad_min (0 = keep all).

Metrics (computed only inside kept non-large boxes, then aggregated per bucket)
-------------------------------------------------------------------------------
* depth: AbsRel(down), SI-RMSE(down), Pearson(up) between generated-image
         estimated depth and the input depth condition.
* seg  : mIoU(up), boundary-F1(up) between generated seg map and condition seg.
* edge : Edge-F1(up), Chamfer(down), Soft-IoU(up) of generated vs reference RGB
         edges (symmetric blurred-Canny), inside each box.

Inputs per method
-----------------
* --gen_image_root/<method>/<stem>.png         generated RGB (for edge + DA3 fallback)
* --gen_depth_root/<method>/depth_npy/<stem>.depth.npy   precomputed DA3 depth (depth mode)
* --gen_seg_root/<method>/<stem>.{npy,png}     generated seg label map  (seg mode)
The script auto-detects the layout; missing/corrupt files are skipped + warned.

Outputs
-------
* <out>/<condition>_nonlarge_fidelity_summary.csv  (one row per method)
* <out>/<condition>_per_object.json                (per-image, per-object debug)
* <out>/debug/<method>/<stem>.png                  (debug panels, capped count)
* prints a markdown table + the CSV path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt, label as cc_label
except Exception:  # pragma: no cover
    distance_transform_edt = None
    cc_label = None


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=["depth", "seg", "edge"])
    p.add_argument("--methods", nargs="+", required=True, help="Method names to evaluate.")

    # YOLO + reference
    p.add_argument("--yolo_dir", required=True,
                   help="run_yoloe_seg output dir (images/<stem>/detections.json).")
    p.add_argument("--ref_image_root", required=True,
                   help="Reference RGB + caption root: <stem>.{jpg,png} (+ .txt).")

    # condition maps
    p.add_argument("--condition_root", required=True,
                   help="depth: <stem>.depth.npy ; seg: <stem>.sam2_label.npy ; edge: <stem>.edge.png")

    # generated outputs (per method)
    p.add_argument("--gen_image_root", default="",
                   help="Root with <method>/<stem>.png generated RGB (edge mode / DA3 fallback).")
    p.add_argument("--gen_depth_root", default="",
                   help="depth mode: root with <method>/depth_npy/<stem>.depth.npy (precomputed DA3).")
    p.add_argument("--gen_seg_root", default="",
                   help="seg mode: root with <method>/<stem>.{npy,png} generated seg label maps.")
    p.add_argument("--gen_edge_root", default="",
                   help="edge mode: precomputed extracted-edge root with "
                        "<method>/pred/<stem>.edge.png and <method>/gt/<stem>.edge.png "
                        "(reuses the edge_consistency symmetric blurred-Canny maps).")

    # DA3 fallback (only if --gen_depth_root not given) -- needs deco env
    p.add_argument("--da3_src", default="t2i/third_party/depth-anything-3/src")
    p.add_argument("--da3_model_dir", default="t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--device", default="cuda:0")

    # canonical frame + selection thresholds
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--small_thr", type=float, default=0.02)
    p.add_argument("--medium_thr", type=float, default=0.10)
    p.add_argument("--min_box_ratio", type=float, default=0.0,
                   help="Drop boxes smaller than this (noise floor).")
    p.add_argument("--min_score", type=float, default=0.0, help="YOLO confidence gate.")
    p.add_argument("--ioa_cond_thr", type=float, default=0.5)
    p.add_argument("--ioa_box_thr", type=float, default=0.3)
    p.add_argument("--edge_min_pixels", type=int, default=20)
    p.add_argument("--edge_box_ratio_thr", type=float, default=0.005)
    p.add_argument("--depth_grad_min", type=float, default=0.0,
                   help="Skip depth boxes with mean cond-depth gradient below this (0=keep all).")

    # metric params
    p.add_argument("--depth_align", choices=["global", "per_box"], default="global",
                   help="global: single per-image affine fit (standard, penalizes wrong "
                        "object placement). per_box: re-fit each box (very forgiving).")
    p.add_argument("--use_polygon_mask", action="store_true",
                   help="Restrict depth/edge metrics to the YOLO instance polygon inside the "
                        "box (object-focused, removes easy background pixels).")
    p.add_argument("--delta_thr", type=float, default=1.25, help="Depth delta1 accuracy ratio.")
    p.add_argument("--boundary_tolerance", type=int, default=2)
    p.add_argument("--absrel_min", type=float, default=0.1)
    p.add_argument("--blur_kernel", type=int, default=11)
    p.add_argument("--canny_low", type=int, default=70)
    p.add_argument("--canny_high", type=int, default=150)
    p.add_argument("--min_box_pixels", type=int, default=16,
                   help="Need at least this many valid pixels in a box to score it.")

    # io
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--debug_images", type=int, default=8,
                   help="Max debug panels per method (0 to disable).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")
DEPTH_EXTS = (".depth.npy", ".npy", ".depth.png", ".png")
SEG_EXTS = (".sam2_pred_label.npy", ".sam2_label.npy", ".npy", ".sam2_label.png", ".png")
EDGE_EXTS = (".edge.png", ".png", ".edge.npy", ".npy")


def warn(msg: str) -> None:
    warnings.warn(msg)
    print(f"[warn] {msg}", flush=True)


def find_file(root: str, stem: str, exts) -> Optional[str]:
    for e in exts:
        p = os.path.join(root, stem + e)
        if os.path.exists(p):
            return p
    return None


def load_array_map(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        with Image.open(path) as im:
            a = np.asarray(im.convert("F") if im.mode in ("I", "I;16", "F") else im.convert("L"),
                           dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    return a


def load_label_map(path: str) -> np.ndarray:
    """Load a seg label map as integer labels."""
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        with Image.open(path) as im:
            a = np.asarray(im.convert("I") if im.mode in ("I", "I;16") else im.convert("L"))
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.int32)


def resize_to(a: np.ndarray, R: int, *, nearest: bool = False) -> np.ndarray:
    if a.shape[:2] == (R, R):
        return a
    if cv2 is None:
        raise RuntimeError("opencv required for resize")
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32), (R, R), interpolation=interp)


def minmax01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return ((a - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def read_orig_size(ref_image_root: str, stem: str) -> Optional[Tuple[int, int]]:
    p = find_file(ref_image_root, stem, IMAGE_EXTS)
    if p is None:
        return None
    try:
        with Image.open(p) as im:
            return im.height, im.width
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# YOLO boxes
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    xyxy: Tuple[float, float, float, float]   # in canonical R frame
    score: float
    label: str
    area_ratio: float
    bucket: str                                # "small" | "medium"
    poly: Optional[np.ndarray] = None          # instance polygon in canonical R frame (Nx2)


def load_boxes(det_path: str, orig_hw: Tuple[int, int], R: int, args) -> List[Box]:
    blob = json.loads(Path(det_path).read_text(encoding="utf-8"))
    H, W = orig_hw
    sx, sy = R / float(W), R / float(H)
    boxes: List[Box] = []
    for d in blob.get("detections", []):
        if float(d.get("score", 0.0)) < args.min_score:
            continue
        bx = d.get("box")
        if not bx or len(bx) != 4:
            continue
        x1, y1, x2, y2 = bx
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1 = max(0.0, min(R, x1)); x2 = max(0.0, min(R, x2))
        y1 = max(0.0, min(R, y1)); y2 = max(0.0, min(R, y2))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratio = area / float(R * R)
        if ratio < args.min_box_ratio or ratio >= args.medium_thr:
            continue  # large or sub-threshold noise
        bucket = "small" if ratio < args.small_thr else "medium"
        poly = d.get("polygon")
        poly_arr = None
        if poly:
            try:
                poly_arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2) * np.array([sx, sy], np.float32)
            except Exception:
                poly_arr = None
        boxes.append(Box((x1, y1, x2, y2), float(d.get("score", 0.0)),
                         str(d.get("label", "?")), ratio, bucket, poly_arr))
    return boxes


def poly_mask(box: Box, R: int) -> Optional[np.ndarray]:
    if box.poly is None or len(box.poly) < 3:
        return None
    m = np.zeros((R, R), dtype=np.uint8)
    cv2.fillPoly(m, [box.poly.astype(np.int32)], 1)
    return m.astype(bool)


def box_slice(box: Box) -> Tuple[slice, slice]:
    x1, y1, x2, y2 = box.xyxy
    xi1, yi1 = int(math.floor(x1)), int(math.floor(y1))
    xi2, yi2 = int(math.ceil(x2)), int(math.ceil(y2))
    return slice(max(0, yi1), max(yi1 + 1, yi2)), slice(max(0, xi1), max(xi1 + 1, xi2))


def box_mask(box: Box, R: int) -> np.ndarray:
    m = np.zeros((R, R), dtype=bool)
    ys, xs = box_slice(box)
    m[ys, xs] = True
    return m


# --------------------------------------------------------------------------- #
# selection rules
# --------------------------------------------------------------------------- #
def seg_components(seg_label: np.ndarray, min_area: int = 16) -> List[np.ndarray]:
    comps = []
    for lab in np.unique(seg_label):
        if lab <= 0:
            continue
        m = seg_label == lab
        if int(m.sum()) >= min_area:
            comps.append(m)
    return comps


def select_seg(box: Box, R: int, comps: List[np.ndarray], args) -> Optional[np.ndarray]:
    """Return the matched component mask (intersected with box) or None."""
    bm = box_mask(box, R)
    barea = float(bm.sum())
    best = None
    best_overlap = 0.0
    for M in comps:
        inter = float(np.logical_and(bm, M).sum())
        if inter <= 0:
            continue
        ioa_cond = inter / max(1.0, float(M.sum()))
        ioa_box = inter / max(1.0, barea)
        if ioa_cond > args.ioa_cond_thr or ioa_box > args.ioa_box_thr:
            if inter > best_overlap:
                best_overlap = inter
                best = np.logical_and(bm, M)
    return best


def select_edge(box: Box, R: int, edge_bin: np.ndarray, args) -> bool:
    bm = box_mask(box, R)
    n = int(np.logical_and(bm, edge_bin).sum())
    barea = max(1.0, float(bm.sum()))
    return n > args.edge_min_pixels or (n / barea) > args.edge_box_ratio_thr


def select_depth(box: Box, R: int, cond_grad: np.ndarray, args) -> bool:
    if args.depth_grad_min <= 0:
        return True
    ys, xs = box_slice(box)
    region = cond_grad[ys, xs]
    return float(region.mean()) >= args.depth_grad_min if region.size else False


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def affine_fit(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    p, g = pred.reshape(-1), gt.reshape(-1)
    n = p.size
    if n < 2:
        return 1.0, 0.0
    sp, sp2, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    det = n * sp2 - sp * sp
    if abs(det) < 1e-8:
        return 1.0, 0.0
    a = (n * spg - sp * sg) / det
    b = (sg - a * sp) / n
    return float(a), float(b)


def depth_box_metrics(pred_aligned: np.ndarray, gt: np.ndarray, absrel_min: float,
                      delta_thr: float = 1.25) -> Dict[str, float]:
    """pred_aligned is already affine-aligned to gt (globally or per-box)."""
    aligned = pred_aligned
    diff = aligned - gt
    si_rmse = float(np.sqrt(np.mean(diff * diff)))
    mask = gt > absrel_min
    denom = max(1, int(mask.sum()))
    if mask.any():
        gtm = np.clip(gt[mask], 1e-3, None)
        am = np.clip(aligned[mask], 1e-3, None)
        absrel = float((np.abs(diff[mask]) / gtm).sum() / denom)
        ratio = np.maximum(am / gtm, gtm / am)
        delta1 = float((ratio < delta_thr).mean())
    else:
        absrel = math.nan; delta1 = math.nan
    pv, gv = aligned.reshape(-1), gt.reshape(-1)
    pv = pv - pv.mean(); gv = gv - gv.mean()
    den = (np.linalg.norm(pv) * np.linalg.norm(gv))
    pear = float((pv * gv).sum() / den) if den > 1e-8 else math.nan
    return {"absrel": absrel, "si_rmse": si_rmse, "delta1": delta1, "pearson": pear}


def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol: int) -> float:
    if distance_transform_edt is None:
        return math.nan
    pe = _mask_boundary(pred_mask)
    ge = _mask_boundary(gt_mask)
    return _tol_f1(pe, ge, tol)


def _mask_boundary(m: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return m
    mm = m.astype(np.uint8)
    er = cv2.erode(mm, np.ones((3, 3), np.uint8), iterations=1)
    return (mm - er).astype(bool)


def _tol_f1(pred: np.ndarray, gt: np.ndarray, tol: int) -> float:
    pn, gn = int(pred.sum()), int(gt.sum())
    if pn == 0 and gn == 0:
        return 1.0
    if pn == 0 or gn == 0:
        return 0.0
    dg = distance_transform_edt(~gt)
    dp = distance_transform_edt(~pred)
    prec = float((dg[pred] <= tol).sum() / pn)
    rec = float((dp[gt] <= tol).sum() / gn)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    return inter / union if union > 0 else math.nan


def chamfer(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    if distance_transform_edt is None:
        return math.nan
    if not pred_bin.any() and not gt_bin.any():
        return 0.0
    if not pred_bin.any() or not gt_bin.any():
        return math.nan
    dg = distance_transform_edt(~gt_bin)
    dp = distance_transform_edt(~pred_bin)
    return float(0.5 * (dg[pred_bin].mean() + dp[gt_bin].mean()))


def blurred_canny(rgb: np.ndarray, k: int, lo: int, hi: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return (cv2.Canny(gray, lo, hi) > 0)


def sobel_mag(a: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


# --------------------------------------------------------------------------- #
# debug rendering
# --------------------------------------------------------------------------- #
def colorize01(a01: np.ndarray) -> np.ndarray:
    u8 = (np.clip(a01, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)  # BGR


def _label_panel(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_boxes(img_bgr: np.ndarray, boxes: List[Box], kept_set: set) -> np.ndarray:
    out = img_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b.xyxy]
        color = {"small": (0, 200, 0), "medium": (0, 165, 255)}.get(b.bucket, (128, 128, 128))
        if i not in kept_set:
            color = (90, 90, 90)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if i in kept_set else 1)
    return out


def save_debug_panel(out_path: Path, panels: List[Tuple[str, np.ndarray]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = [_label_panel(p, t) for t, p in panels]
    strip = np.concatenate(labeled, axis=1)
    cv2.imwrite(str(out_path), strip)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Bucket:
    vals: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, m: Dict[str, float]) -> None:
        for k, v in m.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            self.vals.setdefault(k, []).append(float(v))

    def mean(self, k: str) -> float:
        xs = self.vals.get(k, [])
        return float(np.mean(xs)) if xs else math.nan


def stem_list(yolo_dir: str, max_samples: int) -> List[str]:
    img_dir = Path(yolo_dir) / "images"
    stems = sorted(p.name for p in img_dir.iterdir() if (p / "detections.json").exists())
    if max_samples and max_samples > 0:
        stems = stems[:max_samples]
    return stems


# --------------------------------------------------------------------------- #
# depth path
# --------------------------------------------------------------------------- #
def run_depth(args, stems: List[str]) -> Tuple[Dict[str, Dict[str, Bucket]], Dict]:
    per_object: Dict = {}
    method_buckets: Dict[str, Dict[str, Bucket]] = {
        m: {"small": Bucket(), "medium": Bucket()} for m in args.methods
    }
    counts = {m: {"small": 0, "medium": 0} for m in args.methods}
    R = args.resolution
    da3 = None  # lazy
    dbg_count = {m: 0 for m in args.methods}

    for si, stem in enumerate(stems):
        det = os.path.join(args.yolo_dir, "images", stem, "detections.json")
        if not os.path.exists(det):
            continue
        orig_hw = read_orig_size(args.ref_image_root, stem)
        if orig_hw is None:
            warn(f"{stem}: missing reference image; skip"); continue
        cond_path = find_file(args.condition_root, stem, DEPTH_EXTS)
        if cond_path is None:
            warn(f"{stem}: missing condition depth; skip"); continue
        try:
            cond = minmax01(resize_to(load_array_map(cond_path), R))
        except Exception as e:
            warn(f"{stem}: bad condition depth ({e}); skip"); continue

        boxes = load_boxes(det, orig_hw, R, args)
        if not boxes:
            continue
        cond_grad = minmax01(sobel_mag(cond)) if args.depth_grad_min > 0 else None
        kept = [b for b in boxes if select_depth(b, R, cond_grad, args)]
        if not kept:
            continue

        for method in args.methods:
            gen = _load_gen_depth(args, method, stem, R, da3_getter=lambda: None)
            if gen is None and args.gen_depth_root == "":
                # DA3 fallback
                if da3 is None:
                    da3 = _load_da3(args)
                gen = _da3_depth(da3, args, method, stem, R)
            if gen is None:
                continue
            gen = minmax01(gen)
            # global per-image affine alignment (standard scale-invariant protocol)
            if args.depth_align == "global":
                ga, gb = affine_fit(gen, cond)
                gen_glob = ga * gen + gb
            obj_records = []
            err_frame = np.zeros((R, R), dtype=np.float32)
            for b in kept:
                ys, xs = box_slice(b)
                gp, gt = gen[ys, xs], cond[ys, xs]
                if gp.size < args.min_box_pixels:
                    continue
                if args.depth_align == "global":
                    aligned = gen_glob[ys, xs]
                else:
                    a, bb = affine_fit(gp, gt)
                    aligned = a * gp + bb
                # optional object polygon mask within the box
                sel = None
                if args.use_polygon_mask:
                    pm = poly_mask(b, R)
                    if pm is not None:
                        sel = pm[ys, xs]
                        if int(sel.sum()) < args.min_box_pixels:
                            sel = None
                if sel is not None:
                    aligned_e, gt_e = aligned[sel], gt[sel]
                else:
                    aligned_e, gt_e = aligned, gt
                m = depth_box_metrics(aligned_e, gt_e, args.absrel_min, args.delta_thr)
                err_frame[ys, xs] = np.abs(aligned - gt)
                method_buckets[method][b.bucket].add(m)
                counts[method][b.bucket] += 1
                obj_records.append({"bucket": b.bucket, "label": b.label,
                                    "area_ratio": round(b.area_ratio, 5),
                                    "box": [round(v, 1) for v in b.xyxy], **{k: round(v, 5) for k, v in m.items() if not math.isnan(v)}})
            if obj_records:
                per_object.setdefault(stem, {})[method] = obj_records
                if args.debug_images and dbg_count[method] < args.debug_images:
                    _depth_debug(args, method, stem, R, boxes, kept, cond, gen, err_frame)
                    dbg_count[method] += 1
        if (si + 1) % 100 == 0:
            print(f"  [depth] {si + 1}/{len(stems)} images", flush=True)

    return method_buckets, {"per_object": per_object, "counts": counts}


def _depth_debug(args, method, stem, R, boxes, kept, cond, gen, err_frame) -> None:
    kept_idx = {boxes.index(b) for b in kept}
    ref_p = find_file(args.ref_image_root, stem, IMAGE_EXTS)
    ref = (np.asarray(Image.open(ref_p).convert("RGB"))[:, :, ::-1] if ref_p else np.zeros((R, R, 3), np.uint8))
    ref = resize_to(ref.astype(np.float32), R).astype(np.uint8)
    gen_rgb = None
    if args.gen_image_root:
        gp = os.path.join(args.gen_image_root, method, f"{stem}.png")
        if os.path.exists(gp):
            gen_rgb = resize_to(np.asarray(Image.open(gp).convert("RGB"))[:, :, ::-1].astype(np.float32), R).astype(np.uint8)
    panels = [("ref RGB + boxes", _draw_boxes(ref, boxes, kept_idx))]
    if gen_rgb is not None:
        panels.append(("gen RGB", gen_rgb))
    panels.append(("cond depth", colorize01(cond)))
    panels.append(("gen depth (DA3)", colorize01(gen)))
    panels.append(("|err| in non-large boxes", colorize01(minmax01(err_frame) if err_frame.max() > 0 else err_frame)))
    save_debug_panel(Path(args.output_dir) / "debug" / method / f"{stem}.png", panels)


def _load_gen_depth(args, method: str, stem: str, R: int, da3_getter) -> Optional[np.ndarray]:
    if not args.gen_depth_root:
        return None
    for cand in (os.path.join(args.gen_depth_root, method, "depth_npy", f"{stem}.depth.npy"),
                 os.path.join(args.gen_depth_root, method, "depth_npy", f"{stem}.npy"),
                 os.path.join(args.gen_depth_root, method, f"{stem}.depth.npy")):
        if os.path.exists(cand):
            try:
                return resize_to(load_array_map(cand), R)
            except Exception as e:
                warn(f"{method}/{stem}: bad gen depth ({e})"); return None
    return None


def _load_da3(args):
    import sys, types
    if args.da3_src not in sys.path:
        sys.path.insert(0, args.da3_src)
    es = types.ModuleType("depth_anything_3.utils.export")
    es.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("export off"))
    sys.modules["depth_anything_3.utils.export"] = es
    if "moviepy.editor" not in sys.modules:
        mv = types.ModuleType("moviepy"); ed = types.ModuleType("moviepy.editor")
        ed.ImageSequenceClip = object; mv.editor = ed
        sys.modules.setdefault("moviepy", mv); sys.modules["moviepy.editor"] = ed
    import torch
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(args.da3_model_dir).to(torch.device(args.device)).eval()


def _da3_depth(da3, args, method: str, stem: str, R: int) -> Optional[np.ndarray]:
    import torch
    import torch.nn.functional as F
    p = os.path.join(args.gen_image_root, method, f"{stem}.png")
    if not os.path.exists(p):
        return None
    with torch.no_grad():
        res = da3.inference([p], process_res=504, process_res_method="upper_bound_resize", use_ray_pose=False)
    d = torch.from_numpy(res.depth[0].astype(np.float32))[None, None]
    d = F.interpolate(d, size=(R, R), mode="bilinear", align_corners=False)[0, 0]
    return d.cpu().numpy()


# --------------------------------------------------------------------------- #
# seg path
# --------------------------------------------------------------------------- #
def run_seg(args, stems: List[str]) -> Tuple[Dict[str, Dict[str, Bucket]], Dict]:
    per_object: Dict = {}
    method_buckets = {m: {"small": Bucket(), "medium": Bucket()} for m in args.methods}
    counts = {m: {"small": 0, "medium": 0} for m in args.methods}
    R = args.resolution
    if not args.gen_seg_root:
        raise ValueError("seg mode needs --gen_seg_root with <method>/<stem>.{npy,png}")

    for si, stem in enumerate(stems):
        det = os.path.join(args.yolo_dir, "images", stem, "detections.json")
        orig_hw = read_orig_size(args.ref_image_root, stem)
        cond_path = find_file(args.condition_root, stem, SEG_EXTS)
        if orig_hw is None or cond_path is None:
            warn(f"{stem}: missing ref/cond seg; skip"); continue
        try:
            cond_lab = load_label_map(cond_path)
            cond_lab = resize_to(cond_lab, R, nearest=True).astype(np.int32)
        except Exception as e:
            warn(f"{stem}: bad cond seg ({e}); skip"); continue
        comps = seg_components(cond_lab)
        boxes = load_boxes(det, orig_hw, R, args)
        kept = [(b, m) for b in boxes if (m := select_seg(b, R, comps, args)) is not None]
        if not kept:
            continue
        for method in args.methods:
            gp = find_file(os.path.join(args.gen_seg_root, method), stem, SEG_EXTS)
            if gp is None:
                continue
            try:
                gen_lab = resize_to(load_label_map(gp), R, nearest=True).astype(np.int32)
            except Exception as e:
                warn(f"{method}/{stem}: bad gen seg ({e})"); continue
            recs = []
            for b, comp_mask in kept:
                # GT region = matched condition component within box.
                gt_region = comp_mask
                # Pred region = the generated label that best overlaps gt_region, restricted to box.
                bm = box_mask(b, R)
                pred_region = _best_pred_region(gen_lab, gt_region, bm)
                m = {"miou": iou(pred_region, gt_region),
                     "boundary_f1": boundary_f1(pred_region, gt_region, args.boundary_tolerance)}
                method_buckets[method][b.bucket].add(m)
                counts[method][b.bucket] += 1
                recs.append({"bucket": b.bucket, "label": b.label,
                             "area_ratio": round(b.area_ratio, 5),
                             **{k: round(v, 5) for k, v in m.items() if not math.isnan(v)}})
            if recs:
                per_object.setdefault(stem, {})[method] = recs
        if (si + 1) % 100 == 0:
            print(f"  [seg] {si + 1}/{len(stems)} images", flush=True)
    return method_buckets, {"per_object": per_object, "counts": counts}


def _best_pred_region(gen_lab: np.ndarray, gt_region: np.ndarray, box_m: np.ndarray) -> np.ndarray:
    cand = gen_lab[np.logical_and(gt_region, box_m)]
    cand = cand[cand > 0]
    if cand.size == 0:
        return np.zeros_like(gt_region, dtype=bool)
    vals, cnt = np.unique(cand, return_counts=True)
    best = vals[int(np.argmax(cnt))]
    return np.logical_and(gen_lab == best, box_m)


# --------------------------------------------------------------------------- #
# edge path
# --------------------------------------------------------------------------- #
def run_edge(args, stems: List[str]) -> Tuple[Dict[str, Dict[str, Bucket]], Dict]:
    per_object: Dict = {}
    method_buckets = {m: {"small": Bucket(), "medium": Bucket()} for m in args.methods}
    counts = {m: {"small": 0, "medium": 0} for m in args.methods}
    R = args.resolution
    precomp = bool(args.gen_edge_root)
    if not precomp and not args.gen_image_root:
        raise ValueError("edge mode needs --gen_edge_root (precomputed) or --gen_image_root (RGB)")

    def load_edge_bin(path: str) -> np.ndarray:
        a = load_array_map(path)
        thr = 0.5 * a.max() if a.max() > 1 else 0.5
        return resize_to(a, R) > thr

    for si, stem in enumerate(stems):
        det = os.path.join(args.yolo_dir, "images", stem, "detections.json")
        orig_hw = read_orig_size(args.ref_image_root, stem)
        if orig_hw is None:
            warn(f"{stem}: missing ref RGB size; skip"); continue

        # reference GT edge + selection edge: prefer precomputed symmetric gt edge.
        gt_edge = None
        if precomp:
            for m in args.methods:
                gtp = find_file(os.path.join(args.gen_edge_root, m, "gt"), stem, EDGE_EXTS)
                if gtp is not None:
                    gt_edge = load_edge_bin(gtp); break
        if gt_edge is None:
            ref_p = find_file(args.ref_image_root, stem, IMAGE_EXTS)
            if ref_p is None:
                warn(f"{stem}: no gt edge / ref RGB; skip"); continue
            ref_rgb = resize_to(np.asarray(Image.open(ref_p).convert("RGB")).astype(np.float32), R).astype(np.uint8)
            gt_edge = blurred_canny(ref_rgb, args.blur_kernel, args.canny_low, args.canny_high)
        cond_edge_p = find_file(args.condition_root, stem, EDGE_EXTS) if args.condition_root else None
        cond_bin = load_edge_bin(cond_edge_p) if cond_edge_p else gt_edge

        boxes = load_boxes(det, orig_hw, R, args)
        kept = [b for b in boxes if select_edge(b, R, cond_bin, args)]
        if not kept:
            continue
        for method in args.methods:
            pred_edge = None
            if precomp:
                pp = find_file(os.path.join(args.gen_edge_root, method, "pred"), stem, EDGE_EXTS)
                if pp is not None:
                    try:
                        pred_edge = load_edge_bin(pp)
                    except Exception as e:
                        warn(f"{method}/{stem}: bad pred edge ({e})")
            if pred_edge is None and args.gen_image_root:
                gp = find_file(os.path.join(args.gen_image_root, method), stem, IMAGE_EXTS)
                if gp is not None:
                    try:
                        gen_rgb = resize_to(np.asarray(Image.open(gp).convert("RGB")).astype(np.float32), R).astype(np.uint8)
                        pred_edge = blurred_canny(gen_rgb, args.blur_kernel, args.canny_low, args.canny_high)
                    except Exception as e:
                        warn(f"{method}/{stem}: bad gen RGB ({e})")
            if pred_edge is None:
                continue
            recs = []
            for b in kept:
                ys, xs = box_slice(b)
                pe, ge = pred_edge[ys, xs], gt_edge[ys, xs]
                if args.use_polygon_mask:
                    pm = poly_mask(b, R)
                    if pm is not None:
                        sel = pm[ys, xs]
                        pe, ge = np.logical_and(pe, sel), np.logical_and(ge, sel)
                f1 = _tol_f1(pe, ge, args.boundary_tolerance)
                ch = chamfer(pe, ge)
                inter = float(np.logical_and(pe, ge).sum())
                union = float(np.logical_or(pe, ge).sum())
                m = {"edge_f1": f1, "chamfer": ch,
                     "soft_iou": inter / union if union > 0 else math.nan}
                method_buckets[method][b.bucket].add(m)
                counts[method][b.bucket] += 1
                recs.append({"bucket": b.bucket, "label": b.label,
                             "area_ratio": round(b.area_ratio, 5),
                             **{k: round(v, 5) for k, v in m.items() if not math.isnan(v)}})
            if recs:
                per_object.setdefault(stem, {})[method] = recs
        if (si + 1) % 100 == 0:
            print(f"  [edge] {si + 1}/{len(stems)} images", flush=True)
    return method_buckets, {"per_object": per_object, "counts": counts}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
METRIC_KEYS = {
    "depth": [("depth_absrel", "absrel"), ("depth_si_rmse", "si_rmse"),
              ("depth_delta1", "delta1")],
    "seg": [("seg_miou", "miou"), ("seg_boundary_f1", "boundary_f1")],
    "edge": [("edge_f1", "edge_f1"), ("edge_chamfer", "chamfer")],
}


def write_outputs(args, method_buckets, extra) -> Tuple[str, List[Dict]]:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cond = args.condition
    keys = METRIC_KEYS[cond]
    counts = extra["counts"]

    rows = []
    for method in args.methods:
        b_small, b_med = method_buckets[method]["small"], method_buckets[method]["medium"]
        n_small, n_med = counts[method]["small"], counts[method]["medium"]
        row = {"method": method, "n_small": n_small, "n_medium": n_med,
               "n_nonlarge": n_small + n_med}
        for out_name, mk in keys:
            s = b_small.mean(mk)
            m = b_med.mean(mk)
            # non-large = pooled over both buckets
            pooled = b_small.vals.get(mk, []) + b_med.vals.get(mk, [])
            nl = float(np.mean(pooled)) if pooled else math.nan
            row[f"small_{out_name}"] = s
            row[f"medium_{out_name}"] = m
            row[f"nonlarge_{out_name}"] = nl
        rows.append(row)

    # CSV column order matches the spec
    cols = ["method", "n_small", "n_medium", "n_nonlarge"]
    for out_name, _ in keys:
        cols += [f"small_{out_name}", f"medium_{out_name}", f"nonlarge_{out_name}"]
    csv_path = out / f"{cond}_nonlarge_fidelity_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.5f}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    json_path = out / f"{cond}_per_object.json"
    json_path.write_text(json.dumps({"args": vars(args), "counts": counts,
                                      "per_object": extra["per_object"]}, indent=2))

    # per-object rank-1 win-rate on the primary metric (paired by stem + object index)
    prim_metric, lower_better = {"depth": ("absrel", True), "seg": ("miou", False),
                                 "edge": ("edge_f1", False)}[cond]
    wins = {m: 0 for m in args.methods}
    medians = {m: [] for m in args.methods}
    n_pairs = 0
    for stem, mm in extra["per_object"].items():
        present = [m for m in args.methods if m in mm]
        if len(present) < 2:
            continue
        n_obj = min(len(mm[m]) for m in present)
        for i in range(n_obj):
            vals = {}
            for m in present:
                v = mm[m][i].get(prim_metric)
                if v is not None:
                    vals[m] = v; medians[m].append(v)
            if len(vals) < 2:
                continue
            n_pairs += 1
            best = min(vals.values()) if lower_better else max(vals.values())
            winners = [m for m, v in vals.items() if v == best]
            for m in winners:
                wins[m] += 1.0 / len(winners)
    wr_lines = [f"Per-object rank-1 win-rate on `{prim_metric}` "
                f"({'lower' if lower_better else 'higher'} better), {n_pairs} paired objects:",
                "", "| method | win-rate | median |", "| --- | --- | --- |"]
    for m in args.methods:
        med = float(np.median(medians[m])) if medians[m] else math.nan
        wr_lines.append(f"| {m} | {(wins[m]/max(1,n_pairs)):.3f} | {med:.4f} |")
    (out / f"{cond}_winrate.md").write_text("\n".join(wr_lines) + "\n")
    print("\n" + "\n".join(wr_lines) + "\n")

    # markdown
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(
            (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))) for c in cols) + " |")
    md_text = "\n".join(md)
    (out / f"{cond}_nonlarge_fidelity_summary.md").write_text(md_text + "\n")
    print("\n" + md_text + "\n")
    print(f"[csv]  {csv_path}")
    print(f"[json] {json_path}")
    return str(csv_path), rows


def main() -> None:
    args = parse_args()
    if cv2 is None or distance_transform_edt is None:
        raise RuntimeError("requires opencv-python and scipy")
    stems = stem_list(args.yolo_dir, args.max_samples)
    print(f"[init] condition={args.condition} methods={args.methods} images={len(stems)}")
    runner = {"depth": run_depth, "seg": run_seg, "edge": run_edge}[args.condition]
    method_buckets, extra = runner(args, stems)
    write_outputs(args, method_buckets, extra)


if __name__ == "__main__":
    main()
