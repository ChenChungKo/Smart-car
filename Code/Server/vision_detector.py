#!/usr/bin/env python3
"""Lightweight YOLO-nano ONNX detector for vision cruise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# COCO classes treated as drive-relevant obstacles.
OBSTACLE_CLASS_IDS = {
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
    14,  # bird
    15,  # cat
    16,  # dog
    24,  # backpack
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    41,  # cup
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    62,  # tv
    63,  # laptop
    64,  # mouse
    65,  # remote
    66,  # keyboard
    67,  # cell phone
    73,  # book
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    @property
    def name(self) -> str:
        if 0 <= self.cls_id < len(COCO_NAMES):
            return COCO_NAMES[self.cls_id]
        return str(self.cls_id)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "yolov8n.onnx"


class YoloOnnxDetector:
    """YOLOv8-nano ONNX via onnxruntime (letterbox + NMS)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
    ):
        import onnxruntime as ort

        path = Path(model_path) if model_path else default_model_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {path}. Run: python3 download_yolo_onnx.py"
            )
        self.path = path
        self.conf = conf
        self.iou = iou
        self.imgsz = int(imgsz)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = self.session.get_inputs()[0].type  # e.g. tensor(float) / tensor(float16)
        self.output_names = [o.name for o in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape
        # Prefer model fixed size when present, else constructor imgsz.
        try:
            if isinstance(shape[-1], int) and shape[-1] > 0:
                self.imgsz = int(shape[-1])
        except Exception:
            pass

    def _letterbox(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        size = self.imgsz
        scale = min(size / h, size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        if "float16" in str(self.input_dtype):
            blob = blob.astype(np.float16)
        return blob, scale, left, top

    def detect(self, bgr: np.ndarray, obstacle_only: bool = True) -> list[Detection]:
        if bgr is None or bgr.size == 0:
            return []
        h0, w0 = bgr.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(bgr)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        pred = np.asarray(outs[0], dtype=np.float32)
        # YOLOv8: (1, 84, N) or (1, N, 84). YOLOv5: (1, N, 85) with objectness.
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] in (84, 85) and pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        if pred.shape[1] == 85:
            # YOLOv5: already-sigmoid probabilities in common ONNX exports.
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            cls_ids = np.argmax(cls_scores, axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            confs = obj * cls_conf
            keep = confs >= self.conf
            if obstacle_only:
                keep &= np.isin(cls_ids, list(OBSTACLE_CLASS_IDS))
            sel = np.where(keep)[0]
            for i in sel:
                cx, cy, bw, bh = map(float, pred[i, :4])
                conf = float(confs[i])
                cls_id = int(cls_ids[i])
                x1 = (cx - bw / 2 - pad_x) / scale
                y1 = (cy - bh / 2 - pad_y) / scale
                x2 = (cx + bw / 2 - pad_x) / scale
                y2 = (cy + bh / 2 - pad_y) / scale
                x1 = float(np.clip(x1, 0, w0 - 1))
                y1 = float(np.clip(y1, 0, h0 - 1))
                x2 = float(np.clip(x2, 0, w0 - 1))
                y2 = float(np.clip(y2, 0, h0 - 1))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(cls_id)
        else:
            cls_scores = pred[:, 4:]
            cls_ids = np.argmax(cls_scores, axis=1)
            confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            keep = confs >= self.conf
            if obstacle_only:
                keep &= np.isin(cls_ids, list(OBSTACLE_CLASS_IDS))
            sel = np.where(keep)[0]
            for i in sel:
                cx, cy, bw, bh = map(float, pred[i, :4])
                conf = float(confs[i])
                cls_id = int(cls_ids[i])
                x1 = (cx - bw / 2 - pad_x) / scale
                y1 = (cy - bh / 2 - pad_y) / scale
                x2 = (cx + bw / 2 - pad_x) / scale
                y2 = (cy + bh / 2 - pad_y) / scale
                x1 = float(np.clip(x1, 0, w0 - 1))
                y1 = float(np.clip(y1, 0, h0 - 1))
                x2 = float(np.clip(x2, 0, w0 - 1))
                y2 = float(np.clip(y2, 0, h0 - 1))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(conf)
                class_ids.append(cls_id)

        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        if idxs is None or len(idxs) == 0:
            return []
        flat = np.asarray(idxs).reshape(-1)
        dets = []
        for i in flat[:50]:
            x, y, w, h = boxes[int(i)]
            dets.append(
                Detection(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    conf=scores[int(i)],
                    cls_id=class_ids[int(i)],
                )
            )
        return dets


SCENE_Y0 = 0.08
SCENE_Y1 = 0.86


def fisheye_circle_mask(shape_hw, margin: float = 0.02) -> np.ndarray:
    h, w = shape_hw[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(min(h, w) * (0.5 - margin))
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return mask


def _close_1d_gaps(flags: np.ndarray, gap: int = 3) -> np.ndarray:
    closed = flags.astype(bool).copy()
    n = len(closed)
    i = 0
    while i < n:
        if closed[i]:
            i += 1
            continue
        j = i
        while j < n and not closed[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= gap:
            closed[i:j] = True
        i = j
    return closed


def _lowest_run_end(flags: np.ndarray, min_len: int = 8) -> int | None:
    """Bottom of the lowest long True-run. Prefer wall foot over a brighter foam top."""
    best_end = None
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if j - i >= min_len:
            end = j - 1
            if best_end is None or end > best_end:
                best_end = end
        i = j
    return best_end


def _foam_mask(bch: np.ndarray, gch: np.ndarray, rch: np.ndarray) -> np.ndarray:
    """Lab foam: CSI is saturated cyan; gimbal USB AWB makes the same foam pale grey-blue."""
    b = bch.astype(np.float32)
    g = gch.astype(np.float32)
    r = rch.astype(np.float32)
    bg = b - g
    cyan = (bg > 8.0) & (b > 120.0) & (r < 110.0)
    pale = (bg > 5.0) & (b > 80.0) & ((b - r) > 0.0)
    return cyan | pale


def _last_bg_drop(row_bg: np.ndarray, lo_frac: float = 0.18, hi_frac: float = 0.94) -> int | None:
    """Last blue-minus-green drop: foam body to floor, not the foam top."""
    if row_bg.size < 12:
        return None
    drop = np.diff(row_bg)
    lo = int(len(row_bg) * lo_frac)
    hi = min(int(len(row_bg) * hi_frac), len(drop))
    last = None
    for j in range(lo, hi):
        if drop[j] <= -5.0 and row_bg[j] > 3.0:
            last = j
    return last


def find_wall_floor_junction(
    bgr: np.ndarray, x0: int | None = None, x1: int | None = None
) -> dict | None:
    """Nearest wall–floor contact in a vertical strip.

    Lab foam is bright cyan (B>G, almost no red). The cutting mat is teal-green
    (B≈G). Take the bottom of the largest cyan blob, not the first HSV match
    and not the bottom of the search box.
    """
    h, w = bgr.shape[:2]
    if x0 is None:
        x0 = w // 3
    if x1 is None:
        x1 = 2 * w // 3
    y0, y1 = int(h * SCENE_Y0), int(h * SCENE_Y1)
    strip = bgr[y0:y1, x0:x1]
    if strip.size < 32:
        return None
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    n = gray.shape[0]
    xs = (x0 + x1) // 2
    bch, gch, rch = cv2.split(strip)
    row_b = bch.mean(axis=1).astype(np.float32)
    row_g = gch.mean(axis=1).astype(np.float32)
    row_r = rch.mean(axis=1).astype(np.float32)
    row_bg = row_b - row_g

    foam = _close_1d_gaps(_foam_mask(row_b, row_g, row_r), gap=8)
    wall_bottom = _lowest_run_end(foam, min_len=12)
    j_drop = _last_bg_drop(row_bg)
    if j_drop is not None:
        if wall_bottom is None:
            wall_bottom = j_drop
        elif wall_bottom <= j_drop <= wall_bottom + 8:
            wall_bottom = j_drop

    if wall_bottom is not None and wall_bottom >= 2:
        return {
            "x": xs,
            "y": y0 + int(wall_bottom),
            "strength": float(np.clip(row_bg[min(wall_bottom, n - 1)] / 60.0, 0.2, 1.0)),
            "frac": int(wall_bottom) / max(n - 1, 1),
            "kind": "wall_bottom",
        }

    row = gray.mean(axis=1).astype(np.float32)
    delta = np.abs(np.diff(row))
    if delta.size < 8:
        return None
    lo, hi = int(n * 0.35), int(n * 0.94)
    sl = delta[lo: min(hi, delta.size)]
    if sl.size < 4:
        return None
    thresh = max(0.08 * 255.0, float(np.percentile(sl, 75)))
    strong = np.flatnonzero(sl >= thresh)
    if strong.size == 0:
        return None
    j = lo + int(strong[-1])
    return {
        "x": xs,
        "y": y0 + j,
        "strength": float(delta[j]) / 255.0,
        "frac": j / max(n - 1, 1),
        "kind": "foot_edge",
    }


def _cyan_x_stats(
    bgr: np.ndarray, y: int, half: int = 10, circle: bool = True
) -> tuple[float | None, float | None, float | None]:
    """Return (median_x, min_x, max_x) of foam pixels around row y."""
    h, w = bgr.shape[:2]
    y0, y1 = max(0, int(y) - half), min(h, int(y) + half + 1)
    roi = bgr[y0:y1]
    if roi.size < 32:
        return None, None, None
    bch, gch, rch = cv2.split(roi)
    foam = _foam_mask(bch, gch, rch)
    if circle:
        foam = foam & (fisheye_circle_mask(bgr.shape)[y0:y1] > 0)
    _ys, xs = np.where(foam)
    if xs.size < 25:
        return None, None, None
    xs = xs.astype(np.float32)
    return float(np.median(xs)), float(xs.min()), float(xs.max())


def wall_aim_point(bgr: np.ndarray, circle: bool = True) -> dict | None:
    """Nearest wall foot plus heading x for azimuth (not centimetres).

    A wall that still crosses the image centre is measured straight ahead
    (closest face). Only a one-sided blob yaws the ultrasonic.
    """
    h, w = bgr.shape[:2]
    bands = (
        ("left", 0, w // 3),
        ("mid", w // 3, 2 * w // 3),
        ("right", 2 * w // 3, w),
    )
    found: list[dict] = []
    for name, x0, x1 in bands:
        hit = find_wall_floor_junction(bgr, x0, x1)
        if hit is None:
            continue
        item = dict(hit)
        item["band"] = name
        found.append(item)
    if not found:
        return None
    mid = next((item for item in found if item["band"] == "mid"), None)
    if mid is None:
        chosen = max(found, key=lambda item: int(item["y"]))
    else:
        chosen = mid
        for item in found:
            if item is mid:
                continue
            nearer = int(item["y"]) >= int(mid["y"]) + 18
            strong = item.get("kind") == "wall_bottom" and float(item.get("strength") or 0) >= 0.35
            if nearer and strong and int(item["y"]) > int(chosen["y"]):
                chosen = item
    mid_x, x_lo, x_hi = _cyan_x_stats(bgr, int(chosen["y"]), circle=circle)
    if mid_x is not None and x_lo is not None and x_hi is not None:
        if x_lo <= w * 0.45 and x_hi >= w * 0.55:
            aim_x = w / 2.0
        else:
            aim_x = mid_x
    else:
        aim_x = float(chosen["x"])
    chosen = dict(chosen)
    chosen["x"] = int(round(aim_x))
    chosen["kind"] = "aim"
    return chosen


def classic_corridor_score(bgr: np.ndarray) -> dict[str, float]:
    """Backup wall/obstacle score in L/M/R. Higher = more blocked.

    Front CSI sits low with the hood filling the bottom of the frame; the
    usable scene is the upper/mid part of the fisheye circle, not the bottom.

    Distance cue: wall–floor junction near the *bottom* of the ROI means little
    floor left (near). Junction high + bright lower third means wall is still far,
    so do not treat colorful distant walls as blocked.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = fisheye_circle_mask(bgr.shape)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    y0 = int(h * SCENE_Y0)
    y1 = int(h * SCENE_Y1)
    roi = gray[y0:y1, :]
    roi_mask = mask[y0:y1, :]
    edges = cv2.Canny(roi, 25, 90)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    lap = cv2.Laplacian(roi, cv2.CV_32F)
    lap = np.abs(lap)
    lap_u8 = np.clip(lap, 0, 255).astype(np.uint8)
    lap_u8 = cv2.bitwise_and(lap_u8, lap_u8, mask=roi_mask)

    bands_x = {
        "left": (0, w // 3),
        "mid": (w // 3, 2 * w // 3),
        "right": (2 * w // 3, w),
    }
    out = {"left": 0.0, "mid": 0.0, "right": 0.0}
    for name, (x0, x1) in bands_x.items():
        e = edges[:, x0:x1]
        m = roi_mask[:, x0:x1]
        valid = max(int(m.sum()) / 255.0, 1.0)
        edge_dens = float(e.sum()) / (255.0 * valid)
        tex = float(lap_u8[:, x0:x1].sum()) / (255.0 * valid)
        band = roi[:, x0:x1]
        pixels = band[m > 0]
        mean_l = float(pixels.mean()) / 255.0 if pixels.size else 0.5

        row_means = []
        for r in range(band.shape[0]):
            rr = band[r][m[r] > 0]
            row_means.append(float(rr.mean()) if rr.size else mean_l * 255.0)
        row_means = np.asarray(row_means, dtype=np.float32)
        n = len(row_means)
        fill = 0.0
        floorish = 0.5
        junction = find_wall_floor_junction(bgr, x0, x1)
        if n >= 8:
            if junction is not None:
                frac_wall = float(
                    np.clip((junction["y"] - y0) / max(y1 - y0 - 1, 1), 0.0, 1.0)
                )
                fill = frac_wall * min(1.0, 0.35 + 2.0 * junction["strength"])
            else:
                d = np.abs(np.diff(row_means))
                j = int(np.argmax(d))
                strength = float(d[j]) / 255.0
                frac_wall = j / max(n - 1, 1)
                if strength > 0.05:
                    fill = frac_wall * min(1.0, strength * 3.5)

            # Floor in front of a near wall is expected; only the rows below
            # the wall foot count as "still open".
            if junction is not None:
                j_local = max(0, min(n - 1, junction["y"] - y0))
                lower = row_means[j_local:]
            else:
                lower = row_means[2 * n // 3 :]
            floorish = float(np.mean(lower > 145)) if lower.size else 0.5
            if junction is None:
                fill *= max(0.0, 1.0 - 0.85 * floorish)

            if name == "mid" and fill >= 0.40:
                fill = max(fill, 0.50)

        prox = max(fill, 1.0 - floorish)
        edge_term = (1.0 * edge_dens + 0.7 * tex) * (0.25 + 0.75 * prox)
        score = edge_term + 1.5 * fill
        if name == "mid" and fill >= 0.50:
            score = max(score, 0.55)
        if name == "mid" and fill >= 0.70:
            score = max(score, 0.80)
        out[name] = float(min(2.5, score))
    return out


def detections_to_lane_scores(
    dets: list[Detection],
    frame_w: int,
    frame_h: int,
    classic: dict[str, float] | None = None,
    classic_weight: float = 1.0,
) -> dict[str, float]:
    """Map detections to left/mid/right block scores."""
    scores = {"left": 0.0, "mid": 0.0, "right": 0.0}
    # Hood occupies lower frame; obstacles for this mount are higher up.
    y_min = frame_h * 0.08
    y_max = frame_h * 0.62
    area_norm = float(frame_w * frame_h) * 0.08
    for d in dets:
        if d.cy < y_min or d.cy > y_max:
            continue
        weight = min(2.5, d.area / max(area_norm, 1.0)) * d.conf
        nx = d.cx / max(frame_w, 1)
        if nx < 0.33:
            scores["left"] += weight
        elif nx > 0.67:
            scores["right"] += weight
        else:
            scores["mid"] += weight
            if d.area > area_norm:
                scores["left"] += 0.15 * weight
                scores["right"] += 0.15 * weight
    if classic:
        for k in scores:
            scores[k] += classic_weight * classic.get(k, 0.0)
    return scores


def draw_debug(
    bgr: np.ndarray,
    dets: list[Detection],
    scores: dict[str, float],
    action: str,
) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    y0, y1 = int(h * SCENE_Y0), int(h * SCENE_Y1)
    cv2.rectangle(out, (w // 3, y0), (2 * w // 3, y1), (0, 255, 255), 1)
    cv2.line(out, (w // 3, y0), (w // 3, y1), (80, 80, 80), 1)
    cv2.line(out, (2 * w // 3, y0), (2 * w // 3, y1), (80, 80, 80), 1)
    for d in dets:
        cv2.rectangle(
            out,
            (int(d.x1), int(d.y1)),
            (int(d.x2), int(d.y2)),
            (0, 180, 255),
            2,
        )
        cv2.putText(
            out,
            f"{d.name} {d.conf:.2f}",
            (int(d.x1), max(15, int(d.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
    text = (
        f"L={scores.get('left', 0):.2f} M={scores.get('mid', 0):.2f} "
        f"R={scores.get('right', 0):.2f}  {action}"
    )
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out
