"""Imports, constants and Abort shared by the route_follow modules."""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics as st
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from aruco_localizer import (
    CAR_PATH,
    MAP_PATH,
    ArucoLocalizer,
    Pose2D,
    draw_map,
    draw_overlay,
    load_json,
    open_grabbers,
    wrap_rad,
)
from obstacle_map import ObstacleMap, path_gap, shift_to_clear, split_walls, to_world
from pose_kalman import PoseKalman
from vision_detector import classic_corridor_score

SERVER = Path(__file__).resolve().parent
SONIC_PIVOT_M = 0.085  # gimbal pan axis ahead of the car centre (the ultrasonic turns about it)
SONIC_ARM_M = 0.03     # ultrasonic face ahead of the pan axis
SONIC_AHEAD_M = SONIC_PIVOT_M + SONIC_ARM_M  # ultrasonic face ahead of the car centre, facing forward
SONIC_HALF_DEG = 8.0   # assumed half width of the ultrasonic beam; the HC-SR04's is wider
SCAN_STEP_M = 0.15     # longest lane step between two looks with the ultrasonic only
BLIND_GAP_M = 0.24     # anything nearer than this (from the ultrasonic face) is in the camera blind zone: it then reports the
                       # bottom row of the frame, 15-17 cm ahead of the nose on the car, not the distance
DEPART_TOL_DEG = 2.0   # a lane is driven only once the heading is within this of it (4 deg over a metre is 7 cm)
CAM_STEP_M = 0.20      # ... with the camera, which sees the lane 35 cm ahead of the nose
WIDE_PANS = (80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0, -80.0, 0.0)  # one full sweep, far left to far right
CAM_PANS = (-30.0, 30.0, 0.0)  # gimbal angles of one camera look (+ = left); ends facing ahead
HOME_WALL_MARGIN_M = 0.03  # the parking spot keeps the whole body this far inside the walls
BLIND_SLANT_GAP_M = 0.16   # nearer than this to the block, only the straight look is trusted for the strafe distance
SIDE_CLEAR_MIN_M = 0.085  # side ultrasonic reading beside a block: less than this is under 2.5 cm between body and block
PASS_MARGIN_M = 0.08   # the tail clears the block's far face by this much before strafing back (pose drifts a few cm)
WALL_ECHO_M = 0.16     # what the camera sees this close to where the pose puts the wall ahead is the wall
CLEAR_MARGIN_M = 0.05  # PLANNING margin: a detour strafes until the path is clear by this much. It has to hold the
# camera's edge error (2 cm) and the error of the strafes after the last tag fix: with 3 cm, lap 13 (2026-11-03) planned
# a pass 1.5-3 cm above B1 and the right front wheel hit the block's near face.
BLOCK_OWN_M = 0.08  # a point this near the fitted outline of the block being passed is that block
MEASURE_NOISE_M = 0.02  # how far apart two looks put the same block edge
# STOP margin: only something this close to the body's width is in the way. Planning and stopping must not use the same
# number: the plan puts the edge 3 cm off, and a look that reads it 0.5 cm closer would stop the car on the block it is
# passing. The stop margin is the planning margin less the noise.
PATH_MARGIN_M = CLEAR_MARGIN_M - MEASURE_NOISE_M
HOLD_FULL_DEG = 6.0    # gyro heading hold during drives and strafes: this heading error gets HOLD_MAX of the turn power
HOLD_MAX = 0.6         # ... at most this share of turn_pwm is added (the move itself keeps going)
TURN_COAST_S = 0.15    # after a gyro turn stops, wait this long: the car still turns a little, the gyro counts it
EDGE_PAN_DEG = 25.0    # the ultrasonic also looks this far to each side for the edges of the path (15 missed a block 3 cm in)


class Abort(Exception):
    pass


# Settings of the route logic that are not choices for one run. They were command-line options; none was ever changed on
# the car, so they live here with their reasons. route_args.parse_args puts them on `args` (args.stop_cm ...).
TUNING = dict(
    width=1280, height=960,        # CSI camera frame
    arrive_m=0.10,                 # a waypoint is reached within this
    align_deg=12.0,                # turn toward the spot first when the bearing is further off than this
    max_segment_m=0.35,            # longest drive between two localizations
    wall_margin_m=0.04,            # every body corner stays this far from the arena walls
    lane_tol_m=0.02,               # strafe back onto the lane beyond this
    max_jump_m=0.15, max_jump_deg=20.0,  # a tag fix further than this from the dead-reckoned pose is doubted
    home_tol_m=0.03, home_tol_deg=4.0,   # parked at home within this
    lookahead_m=0.15,              # steer at a point this far ahead on the route line
    min_turn_s=0.15, max_turn_s=1.2,     # one turn command lasts between these
    settle_s=0.35,                 # wait after stopping before trusting frames
    fix_wait_s=1.5,                # how long to wait for a tag fix
    max_blind_segments=2,          # segments on dead reckoning before searching for tags
    search_step_deg=30.0,          # step of the turn-in-place search for tags
    stop_cm=18.0,                  # ultrasonic stop distance
    creep_cm=32.0, near_score=0.70,      # front camera corridor score (only with camera_safety)
    camera_safety=False,           # foam walls fill the fisheye, so the corridor score stays off
    front_ir_gate_cm=18.0,         # the front IR counts only when the ultrasonic is nearer than this
    blocked_wait_s=6.0,            # wait for a blocked path to clear
    min_battery_v=7.0,             # a warning only
)
