#!/usr/bin/env python3
"""Drive a waypoint route from arena_map.json using ArUco wall tags.

Stop-and-look: the car stops, localizes from the tags (front + rear CSI),
turns in place toward the next waypoint, drives a short straight segment, and
repeats. Between tag fixes the pose is dead-reckoned from the commanded motion
rates in car_nav.json. Ultrasonic, front IR and the front camera risk score
stop a forward segment at any time.

Dry-run unless --arm is given.
    python3 route_follow.py --route loop            # decisions only
    python3 route_follow.py --route loop --arm      # drive
    python3 route_follow.py --calibrate-motion --arm
"""

from __future__ import annotations

from route_common import *  # noqa: F401,F403
from route_args import parse_args
from route_motion import MotionMixin
from route_localization import LocalizationMixin
from route_perception import PerceptionMixin
from route_detour import DetourMixin
from route_plan import PlanMixin
from route_output import OutputMixin


class RouteFollower(MotionMixin, LocalizationMixin, PerceptionMixin, DetourMixin, PlanMixin, OutputMixin):
    def __init__(self, args):
        self.args = args
        self.arena = load_json(Path(args.map))
        self.car_path = Path(args.car)
        self.car = load_json(self.car_path)
        self.motion = self.car["motion"]
        self.localizer = ArucoLocalizer(self.arena, self.car)
        routes = self.arena.get("routes", {})
        if not args.calibrate_motion and args.route not in routes:
            raise SystemExit(f"route '{args.route}' not in {args.map}; have {sorted(routes)}")
        self.route = [tuple(map(float, wp)) for wp in routes.get(args.route, [])]
        self.grabbers = {}
        self.sensors = None
        self.motor = None
        self.servo = None
        self.home_pan, self.home_tilt = 90, 90
        self.right_90_pan, self.left_90_pan = 180, 0
        self.gyro = None                      # imu.YawGyro: the heading change of every move, from the z gyro
        self._pose: Pose2D | None = None
        self.trail: list[tuple[float, float, float]] = []
        self.stop_event = threading.Event()
        self.sonic_misses = 0
        self.travel = 0.0  # forward distance of the move in progress, not yet in the pose
        self.strafe_yaw = 0.0  # rad the car turns per metre of strafe to the left (learned; negative = clockwise)
        self.strafe_acc: list[float] | None = None  # [yaw at the start, metres strafed (left +)] since the last fix
        self.turn_scale = {"left": 1.0, "right": 1.0}  # learned from tag fixes after turns (battery changes the rate)
        self.last_turn: tuple[str, float, float] | None = None  # (kind, yaw before, commanded turn rad), for learning
        self.closing_cm: float | None = None  # ultrasonic stop while closing in on an obstacle to go around
        self.closing_ir_after: float | None = None  # while closing in: the front IR may light after this much travel
        self.closed_in: float | None = None   # gap (m, from the ultrasonic face) expected after the last closing move
        self.detouring = False                # True while going around an obstacle (tag headings are trusted)
        self.use_filter = bool(args.hug and args.pose_filter)
        self.kalman = PoseKalman(*(float(v) for v in self.arena["arena_size_m"]),
                                 tuple(float(v) / 2.0 for v in self.car.get("body_m", [0.29, 0.18])),
                                 float(self.motion.get("strafe_yaw_deg_per_cm", 0.6)))
        self.scanner = None                   # camera obstacle finder (foot_cam.CameraScanner) or None
        self.block_long = float(max(self.arena.get("obstacle_block_m", [0.20, 0.10])))  # longest side of an obstacle block
        self.obstacles = ObstacleMap(self.arena["arena_size_m"], self.arena.get("obstacle_block_m", (0.20, 0.10)))
        self.last_views: list = []
        self.look_no = 0
        self.front_risk = 0.0
        self.debug_dir = SERVER / args.debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if args.save_looks:  # a fresh run keeps only its own looks
            import shutil

            shutil.rmtree(self.debug_dir / "looks", ignore_errors=True)
        self.window = None
        self.writer = None
        self.record_path = Path(args.record) if args.record else None
        if not args.no_window:
            try:
                cv2.namedWindow("route", cv2.WINDOW_NORMAL)
                self.window = "route"
            except cv2.error as exc:
                print(f"WARNING no display for the trajectory window: {exc}", flush=True)
        length, width = (float(v) for v in self.car.get("body_m", [0.29, 0.18]))
        self.body_half = (length / 2.0, width / 2.0)
        self.home: Pose2D | None = None
        self.strafe_mps = float(self.motion.get("strafe_mps", 0.6 * float(self.motion["forward_mps"])))
        self.strafe_base = self.strafe_mps
        self.strafe_flip = False
        self.strafe_wrong = 0
        self.jumps = 0
        self.jump_pose: Pose2D | None = None
        self.parking = False
        self.wall_home = False  # the home is the map's, parked by the wall at the right (park_by_wall)
        self.home_open_loop = False  # the home fix was not usable: park by a relative move from the lane start
        self.lane_yaw: float | None = None
        self.lane_check = False
        self.allow_gap: float | None = None
        self.hug_poses: list[tuple[float, float, float]] = []
        if args.hug:
            self.hug_poses = self.build_hug_poses()
            self.route = [(x, y) for x, y, _ in self.hug_poses] + [self.hug_poses[0][:2]]

    @property
    def pose(self) -> Pose2D | None:
        return self._pose

    @pose.setter
    def pose(self, value: Pose2D | None) -> None:
        """A new pose (a measurement, a fix) has already seen the turning so far: the gyro starts counting again."""
        self._pose = value
        if self.gyro is not None:
            self.gyro.mark()

    # ---- hardware -------------------------------------------------------

    def start(self):
        # Hardware modules load here, so the simulators (route_sim.py, Webots) can import this file anywhere.
        from full_car_test import estimate_battery_percent
        from safe_surround_cruise import SafetySensors, read_battery_voltage
        from servo import Servo

        try:
            from servo import load_gimbal_home, load_left_90_pan, load_right_90_pan

            self.servo = Servo()  # applies gimbal_home.json: ultrasonic straight ahead
            self.home_pan, self.home_tilt = load_gimbal_home()
            self.right_90_pan, self.left_90_pan = load_right_90_pan(self.home_pan), load_left_90_pan()
        except Exception as exc:
            print(f"WARNING gimbal: {exc}", flush=True)
        self.grabbers = open_grabbers(self.car, self.args.width, self.args.height, rear=not self.args.no_rear)
        self.sensors = SafetySensors(
            SimpleNamespace(no_ir=self.args.no_ir, no_sonic=self.args.no_sonic, active_high=self.args.active_high)
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not all(g.latest()[0] is not None for g in self.grabbers.values()):
            time.sleep(0.1)
        missing = [name for name, g in self.grabbers.items() if g.latest()[0] is None]
        if missing:
            raise Abort(f"cameras not ready: {missing}")
        if self.args.hug and self.args.detour and self.args.camera_obstacles and self.servo is not None:
            try:  # one USB camera (the gimbal's) at a time works; the side USB cameras stay closed
                from foot_cam import CameraScanner, FootCamera, GimbalFeed

                feed = GimbalFeed().start()
                feed.wait_ready()
                self.scanner = CameraScanner(FootCamera(self.car), feed)
                print("obstacles: gimbal camera foot lines + ultrasonic + IR", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING gimbal camera unavailable ({exc}); obstacles from the ultrasonic and IR only",
                      flush=True)
        self.start_gyro()
        if self.args.arm:
            battery = read_battery_voltage()
            if battery is None:
                print("WARNING battery voltage unreadable; starting anyway", flush=True)
            else:
                low = f" (below {self.args.min_battery_v}V: motors run slower than calibrated)" \
                    if battery < self.args.min_battery_v else ""
                print(f"ARMED battery={battery:.2f}V ({estimate_battery_percent(battery)}%){low}. "
                      f"Starting in 3 s.", flush=True)
            for n in (3, 2, 1):
                print(n, flush=True)
                if self.stop_event.wait(1.0):
                    raise Abort("stopped")
            from motor import Ordinary_Car

            self.motor = Ordinary_Car()
        else:
            print("DRY-RUN: no motor object; decisions are printed only.", flush=True)

    def start_gyro(self) -> None:
        """The z gyro of the GY-95T, bias measured now (the car stands still while it starts)."""
        if not getattr(self.args, "imu", True) or "imu" not in self.car:
            return
        try:
            from imu import Gy95t, YawGyro

            imu = Gy95t(self.car["imu"].get("port")).start()
            gyro = YawGyro(imu, scale=float(self.car["imu"].get("gyro_z_scale", 1.0)))
            deadline = time.monotonic() + 3.0
            while not gyro.fresh() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not gyro.fresh():
                imu.close()
                print("WARNING IMU sends nothing; the heading between tag fixes comes from the turn rates", flush=True)
                return
            bias, std = gyro.calibrate(2.0)
            if std > 0.3:
                imu.close()
                print(f"WARNING IMU not still while measuring its bias (noise {std:.2f} deg/s); not used", flush=True)
                return
            self.gyro = gyro
            print(f"IMU gyro z: bias {bias:+.3f} deg/s (noise {std:.3f}), scale {gyro.scale:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING IMU unavailable ({exc}); the heading between tag fixes comes from the turn rates", flush=True)

    def request_stop(self, *_args):
        self.stop_event.set()
        self.stop()

    def close(self):
        self.stop_event.set()
        self.stop()
        if self.args.save_learned and self.motor is not None and abs(self.strafe_mps - self.strafe_base) > 0.005:
            self.motion["strafe_mps"] = round(self.strafe_mps, 3)
            self.car["motion"] = self.motion
            self.car_path.write_text(json.dumps(self.car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"saved learned strafe speed {self.strafe_mps:.3f} m/s to {self.car_path.name}", flush=True)
        if self.motor is not None:
            try:
                self.motor.close()
            except Exception:
                pass
        if self.scanner is not None:
            try:
                self.scanner.close()
            except Exception:
                pass
        if self.obstacles.blocks:
            (self.debug_dir / "obstacles.json").write_text(
                json.dumps(self.obstacles.as_dict(), indent=2) + "\n", encoding="utf-8")
        if self.servo is not None and hasattr(self.servo, "release_gimbal"):
            try:  # face ahead, then stop the pulses so the gimbal does not buzz after the run
                self.servo.set_pan(self.home_pan)
                time.sleep(0.4)
                self.servo.release_gimbal()
            except Exception:
                pass
        if self.sensors is not None:
            self.sensors.close()
        if self.gyro is not None:
            self.gyro.imu.close()
        for grabber in self.grabbers.values():
            grabber.close()
        self.save_map()
        if self.writer is not None:
            self.writer.release()
            print(f"saved recording {self.record_path}", flush=True)
        if self.window is not None:
            cv2.destroyWindow(self.window)


def main():
    args = parse_args()
    follower = RouteFollower(args)
    signal.signal(signal.SIGINT, follower.request_stop)
    signal.signal(signal.SIGTERM, follower.request_stop)
    if not follower.motion.get("calibrated") and not args.calibrate_motion:
        print("NOTE: car_nav.json motion is not calibrated; run --calibrate-motion --arm first.", flush=True)
    try:
        follower.start()
        if args.dashboard:
            from dashboard.embed import attach
            attach(follower, args.dashboard)
        if args.calibrate_motion:
            follower.calibrate_motion()
        else:
            follower.follow()
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        follower.close()
        print(f"Stopped; motors off. Map: {follower.debug_dir / 'trajectory.png'}", flush=True)


if __name__ == "__main__":
    main()
