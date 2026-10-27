#!/usr/bin/env python3
"""Solve where each camera is mounted on the car from a static survey (tools/static_survey.py).

    python3 tools/solve_mounts.py 校正1                # all cameras
    python3 tools/solve_mounts.py 校正1 right left     # only these

At every survey pose the car pose (your ruler numbers) and the walls' tags are known. A tag corner seen by a camera is
a ray; the camera's mounting (x, y, z, yaw, pitch on the car, the model of car_nav.json camera_mounts) is whatever
makes the known tag corners fall on the seen rays at all poses: a robust least-squares fit of the ray directions.

Per camera it prints the mount now and solved, the ray error before and after, and the car position that camera's
single tags give at each pose, before and after, and with the pose held out of the fit (leave-one-pose-out: the number
that tells whether the solved mount is real or fitted to the noise). Nothing is written to car_nav.json; the proposal
goes to <survey>/solved_mounts.json.

The 'gimbal' camera is solved at the pan the survey used (straight ahead): its mount is that of the camera on the
gimbal at pan home, not a fixed mount.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, car_from_camera, load_json  # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
FOCAL_PX = 400.0   # ray error -> pixels (a pinhole of 400 px focal length)
GIMBAL_START = {"x_m": 0.10, "y_m": 0.0, "z_m": 0.10, "yaw_deg": 0.0, "pitch_deg": -1.5}
NAMES = ("x_m", "y_m", "z_m", "yaw_deg", "pitch_deg")


def world_T_car(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def load_survey(folder: Path):
    poses = []
    for d in sorted(folder.glob("*_x*")):
        f = d / "obs.json"
        if f.exists():
            o = json.loads(f.read_text())
            poses.append((d.name, o))
    return poses


def observations(poses, camera):
    """[(pose index, tag id, normalised corners 4x2)] from the first frame that has the tag."""
    out = []
    for k, (_name, o) in enumerate(poses):
        cam = o["cameras"].get(camera)
        if not cam:
            continue
        seen = {}
        for frame in cam["frames"]:
            for tag_id, v in frame.items():
                seen.setdefault(int(tag_id), np.array(v["norm"], float))
        out += [(k, t, n) for t, n in seen.items()]
    return out


def ray_residuals(params, mount0, obs, poses, loc):
    mount = dict(mount0)
    mount.update(dict(zip(NAMES, params)))
    cam_T_car = np.linalg.inv(car_from_camera(mount))
    res = []
    for k, tag_id, norm in obs:
        x, y, yaw = poses[k][1]["pose_cm_deg"]
        T = np.linalg.inv(world_T_car(x / 100.0, y / 100.0, math.radians(yaw))) @ np.column_stack(
            [loc.world_corners[tag_id], np.ones(4)]).T
        pred = (cam_T_car @ T)[:3].T
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        meas = np.column_stack([norm, np.ones(4)])
        meas /= np.linalg.norm(meas, axis=1, keepdims=True)
        res.append(((pred - meas) * FOCAL_PX).ravel())
    return np.concatenate(res) if res else np.zeros(1)


def fit(mount0, obs, poses, loc):
    start = np.array([mount0[n] for n in NAMES], float)
    lo = start - np.array([0.25, 0.25, 0.12, 120.0, 45.0])
    hi = start + np.array([0.25, 0.25, 0.12, 120.0, 45.0])
    lo[2], hi[2] = max(lo[2], 0.0), min(hi[2], 0.25)
    best = None
    for dyaw in (-90.0, -45.0, 0.0, 45.0, 90.0):   # the yaw of a badly known camera can be far off: several starts
        x0 = start.copy()
        x0[3] += dyaw
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        r = least_squares(ray_residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0,
                          args=(mount0, obs, poses, loc))
        if best is None or r.cost < best.cost:
            best = r
    return best


def robust_fit(mount0, obs, poses, loc, limit_px: float = 25.0):
    """Fit, drop the tag views that stay further than limit_px from their rays (a torn frame, a wrong id), fit again."""
    kept = list(obs)
    for _ in range(3):
        r = fit(mount0, kept, poses, loc)
        bad = [o for o in kept if np.sqrt(np.mean(ray_residuals(r.x, mount0, [o], poses, loc) ** 2)) > limit_px]
        if not bad or len(kept) - len(bad) < 8:
            break
        kept = [o for o in kept if o not in bad]
    return r, kept


def rms_px(params, mount0, obs, poses, loc) -> float:
    r = ray_residuals(params, mount0, obs, poses, loc)
    return float(np.sqrt(np.mean(r ** 2)))


def pose_errors(loc, camera, mount, obs, poses, pose_indices):
    """Per pose: error of the median single-tag car pose of this camera (cm, cm, deg), using this mount."""
    loc.cam_T_car[camera] = np.linalg.inv(car_from_camera(mount))
    fx = 400.0
    out = {}
    for k in pose_indices:
        fixes = []
        for kk, tag_id, norm in obs:
            if kk == k:
                f = loc._solve(camera, tag_id, norm, fx)
                if f is not None:
                    fixes.append(f)
        if not fixes:
            continue
        x, y, yaw = poses[k][1]["pose_cm_deg"]
        ref = fixes[0].yaw
        dyaw = float(np.median([(f.yaw - ref + math.pi) % (2 * math.pi) - math.pi for f in fixes])) + ref
        ex = float(np.median([f.x for f in fixes])) * 100.0 - x
        ey = float(np.median([f.y for f in fixes])) * 100.0 - y
        eh = math.degrees((dyaw - math.radians(yaw) + math.pi) % (2 * math.pi) - math.pi)
        out[k] = (ex, ey, eh)
    return out


def summary(errs) -> str:
    if not errs:
        return "no fixes"
    a = np.array(list(errs.values()))
    horiz = np.hypot(a[:, 0], a[:, 1])
    return (f"position error mean {horiz.mean():.1f} cm (max {horiz.max():.1f}), "
            f"heading mean {np.abs(a[:, 2]).mean():.1f} deg (max {np.abs(a[:, 2]).max():.1f}), {len(a)} poses")


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "校正1"
    only = sys.argv[2:]
    folder = SERVER / "route_follow_debug" / "survey" / label
    poses = load_survey(folder)
    print(f"{len(poses)} poses in {folder}")
    arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
    loc = ArucoLocalizer(arena, car)
    mounts = {n: dict(m) for n, m in car["camera_mounts"].items()}
    mounts["gimbal"] = dict(GIMBAL_START)
    proposal = {}
    for camera in ("front", "rear", "right", "left", "gimbal"):
        if only and camera not in only:
            continue
        obs = observations(poses, camera)
        indices = sorted({k for k, _t, _n in obs})
        if len(indices) < 4:
            print(f"\n== {camera}: only {len(indices)} poses with tags; skipped")
            continue
        base = {n: float(mounts[camera][n]) for n in NAMES}
        base["camera_num"] = mounts[camera].get("camera_num")
        print(f"\n== {camera}: {len(obs)} tag views at {len(indices)} poses")
        before_px = rms_px([base[n] for n in NAMES], base, obs, poses, loc)
        r, kept = robust_fit(base, obs, poses, loc)
        dropped = sorted({(k + 1, t) for k, t, _n in obs} - {(k + 1, t) for k, t, _n in kept})
        if dropped:
            print(f"   dropped {len(obs) - len(kept)} bad tag views (pose, tag): {dropped}")
        obs = kept
        indices = sorted({k for k, _t, _n in obs})
        solved = dict(base)
        solved.update(dict(zip(NAMES, (float(v) for v in r.x))))
        after_px = rms_px(r.x, base, obs, poses, loc)
        print("   mount      " + "  ".join(f"{n:>9s}" for n in NAMES))
        print("   now        " + "  ".join(f"{base[n]:9.3f}" for n in NAMES))
        print("   solved     " + "  ".join(f"{solved[n]:9.3f}" for n in NAMES))
        print(f"   ray error  {before_px:.1f} px -> {after_px:.1f} px (a 400 px pinhole; the tag corners are about 3-8 px)")
        e_before = pose_errors(loc, camera, base, obs, poses, indices)
        e_after = pose_errors(loc, camera, solved, obs, poses, indices)
        print("   car pose from this camera, now:    " + summary(e_before))
        print("   car pose from this camera, solved: " + summary(e_after))
        held = {}
        for k in indices:
            keep = [o for o in obs if o[0] != k]
            rk, _kept = robust_fit(base, keep, poses, loc)
            m = dict(base)
            m.update(dict(zip(NAMES, (float(v) for v in rk.x))))
            held.update(pose_errors(loc, camera, m, [o for o in obs if o[0] == k], poses, [k]))
        print("   car pose, pose held out of the fit: " + summary(held))
        loc.cam_T_car[camera] = np.linalg.inv(car_from_camera(base))
        proposal[camera] = {n: round(solved[n], 4) for n in NAMES}
    out = folder / "solved_mounts.json"
    out.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nproposal written to {out} (car_nav.json untouched)")


if __name__ == "__main__":
    main()
