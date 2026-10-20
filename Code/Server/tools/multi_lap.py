#!/usr/bin/env python3
"""Run several laps one after the other on the car, each with its own log, and print a table at the end.

    python3 tools/multi_lap.py 5                 # 5 laps, wait for Enter before each (reposition, look at the car)
    python3 tools/multi_lap.py 5 --auto          # no waiting: the car parks at home, the next lap starts from there
    python3 tools/multi_lap.py 5 --keep-going    # carry on after a failed lap (default: stop at the first failure)

Each lap runs gimbal_center.py, then route_follow.py --hug --return-home --arm, with
route_follow_debug/lap<N>/ (run.log, run.mp4, looks/, trajectory.png). results.csv in route_follow_debug/ has one
row per lap. A lap counts as finished when the log has "parked at home" and no ABORT / SIM COLLISION line.
"""
import argparse
import csv
import pathlib
import re
import subprocess
import sys
import time

SERVER = pathlib.Path(__file__).resolve().parents[1]
OUT = SERVER / "route_follow_debug"


def summarize(log: str, seconds: float) -> dict:
    battery = re.search(r"ARMED battery=([\d.]+)V", log)
    parked = re.search(r"parked at home \(off by ([\d.]+)cm, ([+-][\d.]+)deg\)", log)
    abort = re.search(r"^ABORT: (.*)$", log, flags=re.M)
    return dict(
        finished=bool(parked) and not abort,
        battery_v=battery.group(1) if battery else "",
        parked_off_cm=parked.group(1) if parked else "",
        parked_off_deg=parked.group(2) if parked else "",
        detours=len(re.findall(r"back on the lane", log)),
        ignored_fixes=len(re.findall(r"ignoring fix", log)),
        strong_fixes=len(re.findall(r"strong fix", log)),
        heading_from_far_tags=len(re.findall(r"far tags agree on the heading", log)),
        abort=abort.group(1) if abort else "",
        minutes=round(seconds / 60.0, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laps", type=int)
    ap.add_argument("--auto", action="store_true", help="do not wait for Enter between laps")
    ap.add_argument("--keep-going", action="store_true", help="carry on after a lap that did not finish")
    ap.add_argument("--first", type=int, default=1, help="number of the first lap (to add to earlier ones)")
    args = ap.parse_args()
    rows = []
    for n in range(args.first, args.first + args.laps):
        if not args.auto:
            input(f"\nLap {n}: car at its start, pointing ahead, blocks in place? Enter to start, Ctrl+C to stop. ")
        lap_dir = OUT / f"lap{n}"
        lap_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "gimbal_center.py"], cwd=SERVER, check=False)
        started = time.time()
        with open(lap_dir / "run.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "route_follow.py", "--hug", "--return-home", "--arm", "--no-window",
                 "--debug-dir", str(lap_dir.relative_to(SERVER)), "--record", str((lap_dir / "run.mp4").relative_to(SERVER))],
                cwd=SERVER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                print("\nstopped by you", flush=True)
                break
        row = dict(lap=n, **summarize((lap_dir / "run.log").read_text(encoding="utf-8"), time.time() - started))
        rows.append(row)
        print(f"\n=== lap {n}: {'FINISHED' if row['finished'] else 'NOT FINISHED'} "
              f"(battery {row['battery_v']} V, parked off {row['parked_off_cm']} cm) {row['abort']}", flush=True)
        if not row["finished"] and not args.keep_going:
            print("Stopping here: look at the car and the log before another lap (use --keep-going to continue).")
            break
    if rows:
        results = OUT / "results.csv"
        new = not results.exists()
        with open(results, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print("\nlap  result        battery  parked off   detours  ignored  strong  heading-from-tags  minutes")
        for r in rows:
            print(f"{r['lap']:>3}  {'finished' if r['finished'] else 'NOT FINISHED':<12}  {r['battery_v']:>6}V  "
                  f"{r['parked_off_cm']:>5} cm {r['parked_off_deg']:>6}  {r['detours']:>7}  {r['ignored_fixes']:>7}  "
                  f"{r['strong_fixes']:>6}  {r['heading_from_far_tags']:>17}  {r['minutes']:>7}")
        done = sum(r["finished"] for r in rows)
        print(f"\n{done}/{len(rows)} laps finished. Table also appended to {results}")


if __name__ == "__main__":
    main()
