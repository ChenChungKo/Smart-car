#!/usr/bin/env python3
"""Solve where the gimbal camera is at every pan angle from a gimbal survey (tools/gimbal_survey.py).

    python3 tools/solve_gimbal.py gimbal1            # prints the result; writes <survey>/solved_gimbal.json
    python3 tools/solve_gimbal.py gimbal1 --write    # also writes it to car_nav.json "gimbal_tags" (a backup is made)
    python3 tools/solve_gimbal.py gimbal1 --check other   # and tries the solved model on another survey it did not use

The car pose at every survey pose is your ruler measurement and the tags are on the map: the model of gimbal_mount.py is
whatever makes the known tag corners fall on the rays the camera saw, at all poses and pans (robust least squares).
It prints the ray error, the car pose the gimbal's tags give at each survey pose with the solved model, and the same with
that pose left out of the fit: the number that says whether the model is real.
"""
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json  # noqa: E402
from gimbal_mount import NAMES, START, car_T_camera  # noqa: E402

FOCAL_PX = 400.0
LIMITS = dict(pivot_x_m=(0.0, 0.16), pivot_y_m=(-0.04, 0.04), radius_m=(0.0, 0.08), phase_deg=(-90, 90), z_m=(0.05, 0.16),
              scale_left=(0.7, 1.3), scale_right=(0.7, 1.3), yaw0_deg=(-15, 15), pitch_deg=(-15, 15), roll_deg=(-10, 10))


def world_T_car(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def load(folder: Path):
    """[(pose index, pan, tag, normalised corners)], [pose (m, m, rad)], names."""
    views, poses, names = [], [], []
    for d in sorted(folder.glob("*_x*")):
        f = d / "obs.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text())
        x, y, yaw = o["pose_cm_deg"]
        poses.append((x / 100.0, y / 100.0, math.radians(yaw)))
        names.append(d.name)
        for pan, tags in o["pans"].items():
            for tag, v in tags.items():
                views.append((len(poses) - 1, float(pan), int(tag), np.array(v["norm"], float)))
    return views, poses, names


def residuals(x, views, poses, loc):
    p = dict(zip(NAMES, x))
    out = []
    for k, pan, tag, norm in views:
        cam_T_world = np.linalg.inv(world_T_car(*poses[k]) @ car_T_camera(p, pan))
        pts = (cam_T_world @ np.column_stack([loc.world_corners[tag], np.ones(4)]).T)[:3].T
        pred = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        meas = np.column_stack([norm, np.ones(4)])
        meas /= np.linalg.norm(meas, axis=1, keepdims=True)
        out.append(((pred - meas) * FOCAL_PX).ravel())
    return np.concatenate(out)


def fit(views, poses, loc):
    x0 = np.array([START[n] for n in NAMES], float)
    lo, hi = (np.array([LIMITS[n][i] for n in NAMES], float) for i in (0, 1))
    kept = list(views)
    for _ in range(3):
        r = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, args=(kept, poses, loc))
        err = [float(np.sqrt(np.mean(residuals(r.x, [v], poses, loc) ** 2))) for v in kept]
        bad = [v for v, e in zip(kept, err) if e > 25.0]
        if not bad or len(kept) - len(bad) < 12:
            break
        kept = [v for v in kept if not any(v is b for b in bad)]
    return r.x, kept


def pose_from(x, views, loc, start):
    """The car pose that fits these tag views with the model x (the pose is the unknown here)."""
    def res(q):
        return residuals(x, [(0, pan, tag, norm) for _k, pan, tag, norm in views], [(q[0], q[1], q[2])], loc)
    r = least_squares(res, np.array(start, float), loss="soft_l1", f_scale=3.0)
    return r.x


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "gimbal1"
    folder = SERVER / "route_follow_debug" / "survey" / label
    views, poses, names = load(folder)
    loc = ArucoLocalizer(load_json(MAP_PATH), load_json(CAR_PATH))
    views = [v for v in views if v[2] in loc.world_corners]
    print(f"{len(poses)} poses, {len(views)} tag views in {folder}")
    if len(poses) < 2 or len(views) < 12:
        raise SystemExit("too little: survey more poses")
    x, kept = fit(views, poses, loc)
    start = np.array([START[n] for n in NAMES], float)
    rms = lambda xx, vv: float(np.sqrt(np.mean(residuals(xx, vv, poses, loc) ** 2)))
    print(f"dropped {len(views) - len(kept)} bad views; ray error {rms(start, kept):.1f} px before -> {rms(x, kept):.1f} px solved (a 400 px pinhole)")
    for n, v in zip(NAMES, x):
        print(f"  {n:12s} {v:8.4f}   (started at {START[n]})" + ("   AT ITS LIMIT" if min(abs(v - LIMITS[n][0]), abs(v - LIMITS[n][1])) < 1e-4 else ""))
    print("\ncar pose from the gimbal's tags alone, against your ruler (cm, deg):")
    print("pose                    views   solved model: dx dy dheading | this pose left out of the fit: dx dy dheading")
    errs, held = [], []
    for k, name in enumerate(names):
        mine = [v for v in kept if v[0] == k]
        if len(mine) < 2:
            print(f"{name:22s} {len(mine):5d}   too few views")
            continue
        def diff(xx):
            q = pose_from(xx, mine, loc, poses[k])
            return ((q[0] - poses[k][0]) * 100, (q[1] - poses[k][1]) * 100, math.degrees((q[2] - poses[k][2] + math.pi) % (2 * math.pi) - math.pi))
        a = diff(x)
        others = [v for v in kept if v[0] != k]
        b = diff(fit(others, poses, loc)[0]) if len({v[0] for v in others}) >= 2 else (float("nan"),) * 3
        errs.append(a)
        held.append(b)
        print(f"{name:22s} {len(mine):5d}   {a[0]:+5.1f} {a[1]:+5.1f} {a[2]:+5.1f}            |  {b[0]:+5.1f} {b[1]:+5.1f} {b[2]:+5.1f}")
    for label2, e in (("solved", errs), ("left out", held)):
        e = np.array(e)
        if len(e):
            h = np.hypot(e[:, 0], e[:, 1])
            print(f"{label2:9s} position mean {np.nanmean(h):.1f} max {np.nanmax(h):.1f} cm; heading mean {np.nanmean(np.abs(e[:, 2])):.1f} max {np.nanmax(np.abs(e[:, 2])):.1f} deg")
    if "--check" in sys.argv:
        other = sys.argv[sys.argv.index("--check") + 1]
        v2, p2, n2 = load(SERVER / "route_follow_debug" / "survey" / other)
        print(f"\nthe solved model on the survey '{other}' (not used in the fit): dx dy dheading")
        e = []
        for k, name in enumerate(n2):
            mine = [v for v in v2 if v[0] == k and v[2] in loc.world_corners]
            if len(mine) < 2:
                continue
            q = pose_from(x, mine, loc, p2[k])
            e.append(((q[0] - p2[k][0]) * 100, (q[1] - p2[k][1]) * 100, math.degrees((q[2] - p2[k][2] + math.pi) % (2 * math.pi) - math.pi)))
            print(f"{name:22s} {len(mine):5d}   {e[-1][0]:+5.1f} {e[-1][1]:+5.1f} {e[-1][2]:+5.1f}")
        e = np.array(e)
        h = np.hypot(e[:, 0], e[:, 1])
        print(f"position mean {h.mean():.1f} max {h.max():.1f} cm; heading mean {np.abs(e[:, 2]).mean():.1f} max {np.abs(e[:, 2]).max():.1f} deg")
    result = {n: round(float(v), 5) for n, v in zip(NAMES, x)}
    (folder / "solved_gimbal.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwritten to {folder / 'solved_gimbal.json'}")
    if "--write" in sys.argv:
        shutil.copy2(CAR_PATH, str(CAR_PATH) + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        car = json.loads(Path(CAR_PATH).read_text(encoding="utf-8"))
        car["gimbal_tags"] = dict(result, note=f"tools/solve_gimbal.py {label}, {time.strftime('%Y-%m-%d')}")
        Path(CAR_PATH).write_text(json.dumps(car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("written to car_nav.json (gimbal_tags); the old file is kept as a .bak")


if __name__ == "__main__":
    main()
