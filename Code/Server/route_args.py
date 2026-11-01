"""Command line of route_follow.py."""

from __future__ import annotations

import argparse

from route_common import *  # noqa: F401,F403


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="ArUco waypoint route following.")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--car", default=str(CAR_PATH))
    p.add_argument("--route", default="loop", help="Route name in arena_map.json routes.")
    p.add_argument("--repeat", action="store_true", help="Loop the route until Ctrl+C.")
    p.add_argument("--arm", action="store_true", help="Enable motors. Without it nothing moves.")
    p.add_argument("--calibrate-motion", action="store_true",
                   help="Measure forward m/s and turn deg/s with tags, write car_nav.json.")
    p.add_argument("--no-rear", action="store_true", help="Localize with the front CSI only.")
    p.add_argument("--start-at-first", action="store_true",
                   help="Drive to the first waypoint even on a closed route (default: join nearest side).")
    p.add_argument("--return-home", action="store_true",
                   help="Treat the start pose as home: run the closed route from there and park back on it "
                        "(same heading), fine-tuning with mecanum strafes.")
    p.add_argument("--hug", action="store_true",
                   help="Mecanum wall-hugging lap: strafe into a lane --hug-gap-m from each wall, drive along it, "
                        "and turn only at corner spots far enough from both walls (ignores --route).")
    p.add_argument("--hug-gap-m", type=float, default=None,
                   help="Body side to wall gap in --hug lanes. Default: the lanes run through the corner turning "
                        "spots, so the lap is a plain rectangle (about 14 cm). Smaller gaps hug the walls closer "
                        "but step inward at every corner to turn."),
    p.add_argument("--no-detour", dest="detour", action="store_false",
                   help="In --hug lanes, stop at an obstacle instead of strafing around it.")
    p.add_argument("--no-pose-filter", dest="pose_filter", action="store_false",
                   help="In --hug, accept or ignore each tag fix with fixed jump limits instead of the pose filter.")
    p.add_argument("--no-learn-motion", dest="learn_motion", action="store_false",
                   help="Do not learn from tag fixes, during the run, how far the car really turns (rate, and how much "
                        "a strafe turns it). Learned values are for this run only; they are not saved.")
    p.add_argument("--save-learned", action="store_true",
                   help="Write the learned strafe speed back to car_nav.json at the end. Off by default: with a weak "
                        "battery the number drifts from run to run.")
    p.add_argument("--no-save-looks", dest="save_looks", action="store_false",
                   help="Do not save every camera look (frames and pose) under debug-dir/looks for replay.")
    p.add_argument("--no-camera-obstacles", dest="camera_obstacles", action="store_false",
                   help="Find obstacles with the ultrasonic and IR only (the gimbal camera is the default).")
    p.add_argument("--approach-cm", type=float, default=5.0,
                   help="Close in to this far in front of the car's nose before strafing around an obstacle. "
                        "The camera sees the floor only from 13 cm out, so the last stretch is driven by "
                        "dead reckoning from the last look; 3 cm or less risks touching it.")
    p.add_argument("--detour-max-m", type=float, default=0.40,
                   help="Give up when the obstacle is still ahead after strafing this far off the lane.")
    p.add_argument("--home", default=None,
                   help="Known home pose 'x,y,yaw_deg' (overrides arena_map.json \"home\").")
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    p.set_defaults(active_high=False)
    p.add_argument("--ir-active-high", action="store_true", dest="active_high")
    p.add_argument("--dashboard", nargs="?", const=5000, type=int, default=None, metavar="PORT",
                   help="Serve the web dashboard (cameras, map, bird's-eye view, sensors) from this lap; open "
                        "http://<the Pi's address>:PORT (default 5000).")
    p.add_argument("--debug-dir", default="route_follow_debug")
    p.add_argument("--no-window", action="store_true", help="Do not open the live trajectory window.")
    p.add_argument("--record", nargs="?", const="route_follow_debug/run.mp4", default=None,
                   help="Save the live view (trajectory plus both cameras) to this mp4 "
                        "(default route_follow_debug/run.mp4).")
    p.set_defaults(**TUNING)  # settings that are not run choices: route_common.TUNING
    return p.parse_args(argv)
