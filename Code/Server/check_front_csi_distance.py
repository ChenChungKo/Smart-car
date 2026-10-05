#!/usr/bin/env python3
"""Check front CSI distance cues against the ultrasonic sensor.

Cruise uses classic_corridor_score (blockage), not centimetres. This tool
estimates centimetres from the wall–floor junction via multi-point 1/d
fitting (pinhole curve when n>=3) and prints it next to the HC-SR04 reading.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from bev_extrinsic import load_kd
from camera_devices import create_csi_still_configuration, csi_array_to_bgr
from servo import Servo, load_gimbal_home
from ultrasonic import Ultrasonic
from vision_detector import classic_corridor_score, draw_debug, find_wall_floor_junction

SERVER = Path(__file__).resolve().parent
DEBUG_DIR = SERVER / "check_front_csi_debug"
GEOM_PATH = SERVER / "csi_front_geom.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Front CSI distance check (camera_num=1) vs ultrasonic."
    )
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--expect", type=float, default=0.0, help="Tape-measure cm.")
    p.add_argument("--camera-height-cm", type=float, default=12.0)
    p.add_argument(
        "--pitch-deg",
        type=float,
        default=None,
        help="Camera pitch down from horizontal. Default from csi_front_geom.json or 16.",
    )
    p.add_argument(
        "--camera-ahead-cm",
        type=float,
        default=None,
        help="How far the CSI lens is in front of the ultrasonic face. Default from json.",
    )
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-servo", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    return p.parse_args()


def load_geom(
    default_height: float, default_pitch: float | None, default_ahead: float | None
) -> tuple[float, float, float, list]:
    height, pitch, ahead = (
        default_height,
        16.0 if default_pitch is None else default_pitch,
        0.0 if default_ahead is None else default_ahead,
    )
    samples: list[dict] = []
    try:
        data = json.loads(GEOM_PATH.read_text(encoding="utf-8"))
        height = float(data.get("camera_height_cm", height))
        if default_pitch is None:
            pitch = float(data.get("pitch_deg", pitch))
        if default_ahead is None:
            ahead = float(data.get("camera_ahead_cm", ahead))
        samples = list(data.get("samples") or [])
    except Exception:
        pass
    return height, pitch, max(0.0, ahead), samples


def save_geom(height_cm: float, pitch_deg: float, ahead_cm: float, samples: list) -> None:
    model = fit_range_model(samples)
    payload = {
        "camera_height_cm": round(height_cm, 2),
        "pitch_deg": round(pitch_deg, 2),
        "camera_ahead_cm": round(ahead_cm, 2),
        "samples": [
            {"y_norm": round(float(s["y_norm"]), 5), "camera_cm": round(float(s["camera_cm"]), 2)}
            for s in samples
        ],
        "notes": "Press c at 3 distances (near/mid/far). vis interpolates 1/d; n>=3 uses pinhole (a+by)/(1+cy).",
    }
    if model is not None:
        payload["fit_kind"] = model["kind"]
        payload["fit_a"] = round(float(model["a"]), 6)
        payload["fit_b"] = round(float(model["b"]), 6)
        payload["fit_c"] = round(float(model["c"]), 6)
        payload["fit_d_min"] = round(float(model["d_min"]), 1)
        payload["fit_d_max"] = round(float(model["d_max"]), 1)
    GEOM_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sample_arrays(samples: list) -> tuple[np.ndarray, np.ndarray]:
    ys = np.array([float(s["y_norm"]) for s in samples], dtype=np.float64)
    ds = np.array([float(s["camera_cm"]) for s in samples], dtype=np.float64)
    return ys, ds


def fit_range_model(samples: list) -> dict | None:
    """Map undistorted y to camera-forward cm.

    Two points: 1/d = b*y + a (only valid between those distances).
    Three or more: 1/d = (a + b y) / (1 + c y), the pitched-pinhole curve, so
    far distances no longer explode from a two-point horizon.
    """
    if len(samples) < 2:
        return None
    ys, ds = _sample_arrays(samples)
    if float(np.max(ds) - np.min(ds)) < 6.0:
        return None
    inv = 1.0 / np.clip(ds, 1.0, None)
    lin = np.stack([np.ones_like(ys), ys], axis=1)
    ak, _, _, _ = np.linalg.lstsq(lin, inv, rcond=None)
    a0, b0 = float(ak[0]), float(ak[1])
    model = {
        "kind": "linear",
        "a": a0,
        "b": b0,
        "c": 0.0,
        "d_min": float(np.min(ds)),
        "d_max": float(np.max(ds)),
        "y_min": float(np.min(ys)),
        "y_max": float(np.max(ys)),
    }
    if len(samples) < 3:
        return model
    frac = np.stack([np.ones_like(ys), ys, -ys / ds], axis=1)
    abc, _, _, _ = np.linalg.lstsq(frac, inv, rcond=None)
    a, b, c = (float(abc[0]), float(abc[1]), float(abc[2]))
    dens = 1.0 + c * ys
    inv_hat = (a + b * ys) / dens
    if np.any(np.abs(dens) < 0.05) or np.any(inv_hat <= 1e-4):
        return model
    model.update(kind="frac", a=a, b=b, c=c)
    return model


def upsert_sample(samples: list, y_norm: float, camera_cm: float) -> list:
    out = [dict(s) for s in samples]
    for i, s in enumerate(out):
        if abs(float(s["camera_cm"]) - camera_cm) < 4.0:
            out[i] = {"y_norm": y_norm, "camera_cm": camera_cm}
            return out
    out.append({"y_norm": y_norm, "camera_cm": camera_cm})
    out.sort(key=lambda s: float(s["camera_cm"]))
    return out


def _inv_from_model(y_norm: float, model: dict) -> float | None:
    den = 1.0 + float(model["c"]) * y_norm
    if abs(den) < 1e-4:
        return None
    inv = (float(model["a"]) + float(model["b"]) * y_norm) / den
    if inv <= 1e-4:
        return None
    return float(inv)


def dist_from_model(
    y_norm: float, model: dict | None, samples: list
) -> tuple[float | None, bool]:
    """Return (camera_cm, extrapolating). Inside the sample y-span, interpolate 1/d."""
    if model is None:
        return None, False
    ys, ds = _sample_arrays(samples)
    y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
    extrap = bool(y_norm < y_lo - 1e-4 or y_norm > y_hi + 1e-4)
    if not extrap and len(samples) >= 2:
        order = np.argsort(ys)
        inv = float(np.interp(y_norm, ys[order], 1.0 / ds[order]))
    else:
        inv = _inv_from_model(y_norm, model)
        if inv is None:
            return None, extrap
    d = 1.0 / inv
    if d < 1.0 or d > 300.0:
        return None, extrap
    return float(d), extrap


def undistort_norm(x: float, y: float, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    return cv2.fisheye.undistortPoints(pts, k, d)[0, 0]


def visual_distance_cm(y_norm: float, height_cm: float, pitch_rad: float) -> float | None:
    sine, cosine = math.sin(pitch_rad), math.cos(pitch_rad)
    denom = y_norm * cosine + sine
    if denom < 0.02:
        return None
    forward = height_cm * (cosine - y_norm * sine) / denom
    if forward < 1.0 or forward > 300.0:
        return None
    return float(forward)


def fit_pitch_rad(y_norm: float, height_cm: float, dist_cm: float) -> float | None:
    if dist_cm <= 1.0:
        return None
    tan_a = (height_cm - dist_cm * y_norm) / (dist_cm + height_cm * y_norm)
    if tan_a <= 0.0:
        return None
    pitch = math.atan(tan_a)
    if pitch < math.radians(1.0) or pitch > math.radians(40.0):
        return None
    return pitch


def read_sonic_median(sonic: Ultrasonic | None, n: int = 3) -> float | None:
    if sonic is None:
        return None
    values = []
    for _ in range(n):
        cm = sonic.get_distance()
        if cm is not None and 1.0 <= float(cm) <= 300.0:
            values.append(float(cm))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def annotate(
    frame: np.ndarray,
    scores: dict,
    junction: dict | None,
    vis_cm: float | None,
    sonic_cm: float | None,
    ahead_cm: float,
    n_samples: int,
    model: dict | None,
    extrap: bool,
) -> np.ndarray:
    out = draw_debug(frame, [], scores, "front-csi")
    if junction is not None:
        x, y = int(junction["x"]), int(junction["y"])
        cv2.line(out, (frame.shape[1] // 3, y), (2 * frame.shape[1] // 3, y), (0, 0, 255), 2)
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
    sonic_from_vis = None if vis_cm is None else vis_cm + ahead_cm
    lines = [
        f"vis={vis_cm:.0f}cm" if vis_cm is not None else "vis=need 2 dists",
        f"sonic={sonic_cm:.0f}cm" if sonic_cm is not None else "sonic=--",
        f"ahead={ahead_cm:.1f}",
        f"n={n_samples}",
        f"scoreM={scores.get('mid', 0):.2f}",
    ]
    if sonic_from_vis is not None:
        lines.append(f"vis+ahead={sonic_from_vis:.0f}")
        if sonic_cm is not None:
            lines.append(f"err={sonic_from_vis - sonic_cm:+.0f}")
    cv2.putText(
        out,
        "  ".join(lines),
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if model is not None:
        span = f"cal {model['d_min']:.0f}-{model['d_max']:.0f}cm"
        if extrap:
            warn = f"{span}  EXTRAP: press c at this distance"
            color = (0, 80, 255)
        else:
            warn = f"{span}  {model['kind']}"
            color = (0, 220, 0)
        cv2.putText(out, warn, (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "[ ] ahead   c add sample   r reset   s save   q quit",
        (8, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    args = parse_args()
    height_cm, pitch_deg, ahead_cm, samples = load_geom(
        args.camera_height_cm, args.pitch_deg, args.camera_ahead_cm
    )
    pitch_rad = math.radians(pitch_deg)
    model = fit_range_model(samples)
    k, d = load_kd("front")
    k = k.astype(np.float64)
    d = d.reshape(-1, 1)[:4].astype(np.float64)

    if not args.no_servo:
        pan, tilt = load_gimbal_home()
        servo = Servo()
        servo.set_servo_pwm("0", pan)
        servo.set_servo_pwm("1", tilt)
        print(f"Gimbal locked straight-ahead pan={pan} tilt={tilt}")
        time.sleep(0.4)

    sonic = None if args.no_sonic else Ultrasonic()
    from picamera2 import Picamera2

    camera = Picamera2(camera_num=1)
    camera.configure(create_csi_still_configuration(camera, 640, 480))
    camera.start()
    camera.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(0.8)
    for _ in range(6):
        camera.capture_array()

    print("請在近／中／遠各按一次 c（例如 12、25、45 cm）。兩點只能管中間那段，越遠誤差會爆。")
    print("紅線必須在藍牆牆腳。出現 EXTRAP 時不要按 r，就在該距離再按 c。")
    print("ahead=鏡頭比超音波超前的公分，[ ] 調整。")
    span = ""
    if model is not None:
        span = f" cal={model['d_min']:.0f}-{model['d_max']:.0f}cm kind={model['kind']}"
    print(f"ahead={ahead_cm:.1f}cm samples={len(samples)}{span} -> {GEOM_PATH.name}", flush=True)

    show = not args.no_window
    if show:
        try:
            cv2.namedWindow("front-csi-distance", cv2.WINDOW_NORMAL)
        except Exception as exc:
            print(f"無法開視窗（{exc}），改為只印數值。")
            show = False

    period = 1.0 / max(args.hz, 0.5)
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    vis_hist: list[float] = []
    try:
        while True:
            started = time.monotonic()
            frame = csi_array_to_bgr(camera.capture_array())
            if frame is None:
                continue
            scores = classic_corridor_score(frame)
            junction = find_wall_floor_junction(frame)
            vis_cm = None
            y_norm = None
            extrap = False
            if junction is not None:
                y_norm = float(undistort_norm(junction["x"], junction["y"], k, d)[1])
                vis_cm, extrap = dist_from_model(y_norm, model, samples)
            sonic_cm = read_sonic_median(sonic)
            overlay = annotate(
                frame,
                scores,
                junction,
                vis_cm,
                sonic_cm,
                ahead_cm,
                len(samples),
                model,
                extrap,
            )
            extra = f"  ahead={ahead_cm:.1f} n={len(samples)}"
            if vis_cm is not None:
                extra += f"  vis+ahead={vis_cm + ahead_cm:.1f}"
                if sonic_cm is not None:
                    extra += f"  err={vis_cm + ahead_cm - sonic_cm:+.1f}"
                if extrap:
                    extra += "  EXTRAP"
                vis_hist.append(vis_cm)
            print(
                f"vis={vis_cm if vis_cm is None else round(vis_cm,1)}cm "
                f"sonic={sonic_cm if sonic_cm is None else round(sonic_cm,1)}cm "
                f"score M={scores['mid']:.2f}{extra}",
                flush=True,
            )
            if show:
                cv2.imshow("front-csi-distance", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                    path = DEBUG_DIR / time.strftime("front_csi_%H%M%S.jpg")
                    cv2.imwrite(str(path), overlay)
                    print(f"saved {path}", flush=True)
                if key in (ord("["), ord("-")):
                    ahead_cm = max(0.0, ahead_cm - 0.5)
                    save_geom(height_cm, math.degrees(pitch_rad), ahead_cm, samples)
                    print(f"camera ahead={ahead_cm:.1f}cm", flush=True)
                if key in (ord("]"), ord("=")):
                    ahead_cm = min(30.0, ahead_cm + 0.5)
                    save_geom(height_cm, math.degrees(pitch_rad), ahead_cm, samples)
                    print(f"camera ahead={ahead_cm:.1f}cm", flush=True)
                if key == ord("r"):
                    samples = []
                    model = None
                    save_geom(height_cm, math.degrees(pitch_rad), ahead_cm, samples)
                    print("已清除距離取樣。請在近／中／遠各按一次 c。", flush=True)
                if key == ord("c") and sonic_cm is not None and y_norm is not None:
                    camera_dist = sonic_cm - ahead_cm
                    if camera_dist <= 1.0:
                        print("校正失敗：sonic-ahead 太小。")
                    else:
                        samples = upsert_sample(samples, y_norm, camera_dist)
                        model = fit_range_model(samples)
                        save_geom(height_cm, math.degrees(pitch_rad), ahead_cm, samples)
                        dists = ", ".join(f"{float(s['camera_cm']):.0f}" for s in samples)
                        if model is None:
                            print(
                                f"已記錄 {camera_dist:.1f}cm（n={len(samples)} [{dists}]）。"
                                "請把牆移到差至少 8cm 的另一距離再按 c。",
                                flush=True,
                            )
                        elif len(samples) < 3:
                            print(
                                f"已記錄 {camera_dist:.1f}cm（n={len(samples)} [{dists}]）。"
                                "請再在更遠（約 40–50 cm）按一次 c，否則遠距會爆。",
                                flush=True,
                            )
                        else:
                            print(
                                f"已記錄 {camera_dist:.1f}cm（n={len(samples)} [{dists}]），"
                                f"擬合={model['kind']} 範圍 {model['d_min']:.0f}-{model['d_max']:.0f}cm。",
                                flush=True,
                            )
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nEnd of program")
    finally:
        try:
            camera.stop()
            camera.close()
        except Exception:
            pass
        if sonic is not None:
            sonic.close()
        if show:
            cv2.destroyAllWindows()

    if vis_hist:
        vis_hist.sort()
        mid = vis_hist[len(vis_hist) // 2]
        print(
            f"summary vis median={mid:.1f}cm "
            f"min={min(vis_hist):.1f} max={max(vis_hist):.1f} n={len(vis_hist)}"
        )
    print("巡航仍以超音波為距離；相機分數只看有沒有擋住。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
