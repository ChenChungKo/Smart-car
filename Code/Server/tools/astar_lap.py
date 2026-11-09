#!/usr/bin/env python3
"""One lap without a route: only the distance from the walls is given. Dry-run unless --arm.

    python3 tools/astar_lap.py                 # plans the lap from where the car stands; nothing moves
    python3 tools/astar_lap.py --arm           # drives it
    options: --wall-gap 5 (cm, right side of the body to the wall)  --corner 22  --margin 7  --speed 0.16

The four lanes along the walls come from the size of the arena, counter-clockwise, the nose along the lane and the wall on
the right. Every lane is planned with A* (astar.py): the path keeps to the straight lane and leaves it only to go around
a block, then comes back. The car turns on the spot at --corner cm from both walls: it does not fit closer (half its
diagonal is 16 cm), so it moves in from the lane to turn.
The car drives a lane without stopping (tools/astar_drive.py follow) and turns on the spot only at the corners, where
the pose is taken again from the tags. For now the four blocks are where the lap layout has them.
Output: route_follow_debug/astar_lap/<time>/ plan.png, result.png, log.json.
"""
import argparse
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import cv2

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from astar import Grid, plan  # noqa: E402
from astar_drive import LAP_BLOCKS, follow, picture, standing_pose  # noqa: E402
from pose_kalman import wrap  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402
from vio import VioFilter  # noqa: E402
from vio_check import TagLogger  # noqa: E402

CORNER_STAND_S = 1.0
WALL_SLACK_M = 0.01      # the lane itself must be free in the grid: walls keep the gap minus this
WALL_GUARD_M = 0.015     # stops the car when the estimated body is this close to a wall
MAX_START_YAW_DEG = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--wall-gap", type=float, default=5.0, help="cm from the right side of the body to the wall on a lane")
    ap.add_argument("--corner", type=float, default=22.0, help="cm from both walls to the car centre where it turns")
    ap.add_argument("--margin", type=float, default=7.0, help="cm kept between the body and the blocks")
    ap.add_argument("--speed", type=float, default=0.16)
    opts = ap.parse_args()
    out = SERVER / "route_follow_debug" / "astar_lap" / time.strftime("%Y%m%d-%H%M%S")
    car = RouteFollower(parse_args(["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if opts.arm else [])))
    tags, log, trail, reason, shade, legs = None, dict(rows=[], legs=[], corners=[], armed=opts.arm), [], None, None, []
    try:
        car.start()
        tags = TagLogger(car)
        p = standing_pose(tags)
        if p is None:
            raise Abort("no pose from the tags")
        if abs(math.degrees(p.yaw)) > MAX_START_YAW_DEG:
            raise Abort("put the car facing the right wall (heading 0)")
        W, H = (float(v) for v in car.arena["arena_size_m"])
        L = opts.corner / 100.0
        C = car.body_half[1] + opts.wall_gap / 100.0                 # lane line: this far from its wall
        corners = [(L, L), (W - L, L), (W - L, H - L), (L, H - L)]
        lanes = [((L, C), (W - L, C)), ((W - C, L), (W - C, H - L)), ((W - L, H - C), (L, H - C)), ((C, H - L), (C, L))]
        wall_margin = max(0.0, opts.wall_gap / 100.0 - WALL_SLACK_M)
        grids = {}
        def grid(yaw):
            key = round(math.degrees(yaw)) % 180
            if key not in grids:
                grids[key] = Grid((W, H), LAP_BLOCKS, car.body_half[0], car.body_half[1], yaw, opts.margin / 100.0, wall_margin)
            return grids[key]
        first = plan(grid(0.0), (p.x, p.y), corners[0])
        if first is None:
            raise Abort("no path to the first corner")
        legs.append(dict(name="to the first corner", path=first, yaw=0.0, turn=None, guard=None))
        for i in range(4):
            a, b, yaw = corners[i], corners[(i + 1) % 4], wrap(i * math.pi / 2.0)
            path = plan(grid(yaw), a, b, lane=lanes[i])
            if path is None:
                raise Abort(f"no path along lane {i + 1}")
            legs.append(dict(name=f"lane {i + 1}", path=path, yaw=yaw, turn=None if i == 0 else math.pi / 2.0, guard=WALL_GUARD_M, lane=lanes[i]))
        print(f"pose x={p.x * 100:.1f} y={p.y * 100:.1f} yaw={math.degrees(p.yaw):+.1f}; body {opts.wall_gap:.0f} cm from the wall on the lanes, turns at {opts.corner:.0f} cm, {opts.margin:.0f} cm from the blocks", flush=True)
        for leg in legs:
            n = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(leg["path"], leg["path"][1:]))
            print(f"  {leg['name']:20s} heading {math.degrees(leg['yaw']):+4.0f}  {n * 100:4.0f} cm: " +
                  " -> ".join(f"({a[0] * 100:.0f},{a[1] * 100:.0f})" for a in leg["path"]), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        g0, g90 = grid(0.0), grid(math.pi / 2.0)
        shade = SimpleNamespace(blocked=g0.blocked | g90.blocked)
        whole = [q for leg in legs for q in leg["path"]]
        cv2.imwrite(str(out / "plan.png"), picture(shade, (W, H), LAP_BLOCKS, whole))
        print(f"plan picture: {out}/plan.png", flush=True)
        log.update(start=[p.x, p.y, p.yaw], paths=[dict(name=l["name"], yaw=l["yaw"], path=l["path"]) for l in legs])
        if not opts.arm:
            print("DRY-RUN: nothing moved. Add --arm to drive.", flush=True)
            return
        if car.gyro is None:
            raise Abort("no IMU")
        f = VioFilter(p.x, p.y, p.yaw)
        t_start = time.monotonic()
        for leg in legs:
            if leg["turn"] is not None:
                kind = "left" if leg["turn"] > 0 else "right"
                car.move(kind, abs(math.degrees(leg["turn"])) / car.turn_rate(kind), leg["turn"])
                now = standing_pose(tags, CORNER_STAND_S)
                if now is None:
                    raise Abort("no pose from the tags at the corner")
                err = wrap(leg["yaw"] - now.yaw)
                if abs(math.degrees(err)) > 1.5:
                    kind = "left" if err > 0 else "right"
                    car.move(kind, max(abs(math.degrees(err)) / car.turn_rate(kind), 0.05), err)
                    now = standing_pose(tags, CORNER_STAND_S)
                    if now is None:
                        raise Abort("no pose from the tags at the corner")
                old = f.x[3:].copy()
                f = VioFilter(now.x, now.y, now.yaw)
                f.x[3:] = old
                leg["path"] = plan(grid(leg["yaw"]), (now.x, now.y), leg["path"][-1], lane=leg["lane"])   # from where it really stands
                if leg["path"] is None:
                    raise Abort(f"no path along {leg['name']} from where the car stands")
                log["paths"].append(dict(name=leg["name"] + " (from the corner)", yaw=leg["yaw"], path=leg["path"]))
                log["corners"].append([now.x, now.y, now.yaw])
                print(f"corner: x={now.x * 100:.1f} y={now.y * 100:.1f} yaw={math.degrees(now.yaw):+.1f}", flush=True)
            n0 = len(log["rows"])
            why = follow(car, tags, f, leg["path"], leg["path"][-1], leg["yaw"], opts.speed, log, trail, ((W, H), LAP_BLOCKS), leg["guard"])
            car.stop()
            c = [r["cross"] * 100 for r in log["rows"][n0:]]; h = [abs(r["herr"]) for r in log["rows"][n0:]]
            print(f"{leg['name']:20s} {why}: {log['rows'][-1]['t'] - log['rows'][n0]['t']:.1f} s, off the path max {max(c):.1f} cm, heading max {max(h):.1f} deg", flush=True)
            time.sleep(0.3)
        reason = f"lap finished in {time.monotonic() - t_start:.1f} s"
        q = standing_pose(tags)
        if q is not None:
            log["end"] = [q.x, q.y, q.yaw]
            print(f"standing at the end: x={q.x * 100:.1f} y={q.y * 100:.1f} yaw={math.degrees(q.yaw):+.1f}; first corner is {corners[0][0] * 100:.0f},{corners[0][1] * 100:.0f}: "
                  f"{math.hypot(q.x - corners[0][0], q.y - corners[0][1]) * 100:.1f} cm from it", flush=True)
    except Abort as exc:
        reason = str(exc)
        print(f"ABORT: {exc}", flush=True)
    finally:
        car.stop()
        if tags is not None:
            tags.stop.set()
        car.close()
    if log["rows"]:
        log.update(reason=reason, tags=tags.rows)
        (out / "log.json").write_text(json.dumps(log))
        whole = [q for leg in legs for q in leg["path"]]
        cv2.imwrite(str(out / "result.png"), picture(shade, (W, H), LAP_BLOCKS, whole, trail))
        print(f"{reason}. pictures: {out}/result.png", flush=True)


if __name__ == "__main__":
    main()
