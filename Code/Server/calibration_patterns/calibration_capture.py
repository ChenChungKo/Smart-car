#!/usr/bin/env python3
"""Capture chessboard calibration images from one camera."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SERVER = ROOT.parent
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_OUTPUT = ROOT / "captures"

sys.path.insert(0, str(SERVER))
from camera_devices import resolve_usb_capture_index, create_csi_still_configuration


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def parse_video_index(device_text):
    if "/dev/video" in device_text:
        return int(device_text.rsplit("video", 1)[1])
    raise ValueError(f"Unsupported USB device string: {device_text}")


def open_usb(device_index, width, height, warmup, warmup_frames):
    capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{device_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize driver-side frame queueing/latency
    time.sleep(warmup)
    for _ in range(warmup_frames):
        capture.read()
        time.sleep(0.1)
    return capture


def open_csi(camera_num, width, height, warmup, warmup_frames):
    from picamera2 import Picamera2

    camera = Picamera2(camera_num=camera_num)
    config = create_csi_still_configuration(camera, width, height)
    camera.configure(config)
    camera.start()
    camera.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(warmup)
    for _ in range(warmup_frames):
        camera.capture_array()
        time.sleep(0.1)
    return camera


def read_csi(camera):
    return True, cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2BGR)


def detect_board(gray, board_size):
    """Try several OpenCV detectors; steep angles/low light often fail the default finder."""
    import numpy as np

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flag_sets = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
    ]
    for flags in flag_sets:
        ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            return True, corners

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, board_size, flags=sb_flags)
        if ok and corners is not None:
            return True, corners.astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    ok, corners = cv2.findChessboardCorners(
        blurred,
        board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        corners = cv2.cornerSubPix(blurred, corners, (5, 5), (-1, -1), criteria)
        return True, corners
    return False, None


def detect_board_preview(gray, board_size, max_width=960):
    """Fast preview-only detection on a downscaled frame."""
    import numpy as np

    height, width = gray.shape[:2]
    scale = 1.0
    if width > max_width:
        scale = max_width / width
        small = cv2.resize(
            gray,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(small, board_size, flags)
    if not ok and hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(small, board_size)
    if ok and corners is not None:
        if scale != 1.0:
            corners = corners.astype(np.float32) / scale
        return True, corners
    return False, None


def preview_display_frame(frame, max_width=1280):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_status(frame, ok, saved_count, camera_name, batch_no, batch_saved, batch_size, manual=False, step=False, max_count=0):
    if step:
        mode = "ENTER=save  q=quit"
        hint = "Move board, then press ENTER in terminal"
        if max_count > 0:
            mode = f"{saved_count}/{max_count} | ENTER=save  q=quit"
    elif manual:
        mode = "SPACE=save  q=quit"
        hint = "Move board before each SPACE"
    else:
        mode = f"batch {batch_no} ({batch_saved}/{batch_size})"
        hint = "Move board between batches"
    text = f"{camera_name} | total={saved_count} | board={'OK' if ok else 'not found'} | {mode}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(frame, hint, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def next_index(output_dir):
    existing = sorted(output_dir.glob("img_raw*.jpg"))
    if not existing:
        return 0
    numbers = []
    for path in existing:
        stem = path.stem
        if stem.startswith("img_raw") and stem[7:].isdigit():
            numbers.append(int(stem[7:]))
    return max(numbers) + 1 if numbers else 0


def wait_for_next_batch(batch_no):
    while True:
        answer = input(
            f"\nBatch {batch_no} finished. Move the chessboard to a NEW position.\n"
            "Press ENTER for next batch, or type q then ENTER to finish: "
        ).strip().lower()
        if answer in ("q", "quit", "exit"):
            return False
        return True


def capture_batch(read_frame, output_dir, camera_name, board_size, batch_size, batch_interval, batch_no, index):
    batch_saved = 0
    last_save = 0.0
    while batch_saved < batch_size:
        ok_frame, frame = read_frame()
        if not ok_frame or frame is None:
            print("Frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_board, corners = detect_board(gray, board_size)
        display = frame.copy()
        if ok_board:
            cv2.drawChessboardCorners(display, board_size, corners, ok_board)
        draw_status(display, ok_board, index, camera_name, batch_no, batch_saved, batch_size)

        now = time.time()
        if ok_board and now - last_save >= batch_interval:
            path = output_dir / f"img_raw{index}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  Saved {path.name} ({batch_saved + 1}/{batch_size} in this batch)")
            batch_saved += 1
            index += 1
            last_save = now
        time.sleep(0.05)
    return batch_saved, index


def capture_auto_loop(read_frame, release, output_dir, camera_name, board_size, auto_count, auto_interval):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    index = next_index(output_dir)
    last_save = 0.0

    print(f"Auto mode: up to {auto_count} photos, one every {auto_interval}s when board is visible.")
    print("Move the chessboard to a new position every time a photo is saved.")

    while saved_count < auto_count:
        ok_frame, frame = read_frame()
        if not ok_frame or frame is None:
            print("Frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_board, _corners = detect_board(gray, board_size)
        now = time.time()
        if ok_board and now - last_save >= auto_interval:
            path = output_dir / f"img_raw{index}.jpg"
            cv2.imwrite(str(path), frame)
            saved_count += 1
            index += 1
            last_save = now
            print(f"  Saved {path.name} ({saved_count}/{auto_count}) — move board now")
        time.sleep(0.05)

    release()
    print(f"Done. Saved {saved_count} images to {output_dir}")
    return saved_count


def capture_step_loop(read_frame, release, output_dir, camera_name, board_size, preview=True, max_count=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    index = next_index(output_dir)
    window = f"calibration-{camera_name}"
    state = {"frame": None, "ok_board": False, "running": True, "quit": False, "saved_count": 0}
    lock = threading.Lock()

    def preview_loop():
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        last_detect = 0.0
        ok_board = False
        corners = None
        while state["running"]:
            ok_frame, frame = read_frame()
            if not ok_frame or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_detect >= 0.15:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok_board, corners = detect_board_preview(gray, board_size)
                last_detect = now
                with lock:
                    state["frame"] = frame.copy()
                    state["ok_board"] = ok_board
            else:
                with lock:
                    state["frame"] = frame.copy()

            display = frame.copy()
            if ok_board and corners is not None:
                cv2.drawChessboardCorners(display, board_size, corners, ok_board)
            draw_status(
                display,
                ok_board,
                state["saved_count"],
                camera_name,
                1,
                state["saved_count"],
                0,
                step=True,
                max_count=max_count,
            )
            cv2.imshow(window, preview_display_frame(display))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                state["quit"] = True
            time.sleep(0.01)

        cv2.destroyAllWindows()

    print("Step mode: move the board, press ENTER in terminal to save one photo.")
    if max_count > 0:
        print(f"Will stop automatically after {max_count} saved photos.")
    else:
        print("Type q then ENTER to finish.")

    preview_thread = None
    if preview:
        print("Live preview window opened.")
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
        time.sleep(0.5)

    while True:
        if state["quit"]:
            break
        if max_count > 0 and saved_count >= max_count:
            break

        if preview:
            with lock:
                ok_board = state["ok_board"]
        else:
            ok_frame, frame = read_frame()
            if not ok_frame or frame is None:
                print("Frame read failed")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok_board, _corners = detect_board(gray, board_size)
            with lock:
                state["frame"] = frame
                state["ok_board"] = ok_board

        status = "board OK" if ok_board else "board NOT found"
        answer = input(
            f"\nPhoto #{saved_count + 1} | {status} | ENTER=save, q=quit: "
        ).strip().lower()
        if answer in ("q", "quit", "exit"):
            break

        with lock:
            frame = state["frame"]
            ok_board = state["ok_board"]
        if frame is None:
            print("No frame yet — wait for preview, then try again.")
            continue
        if not ok_board:
            print("  Not saved — adjust board until corners are detected, then press ENTER again.")
            continue

        path = output_dir / f"img_raw{index}.jpg"
        cv2.imwrite(str(path), frame.copy())
        print(f"  Saved {path.name} ({saved_count + 1} total)")
        saved_count += 1
        index += 1
        state["saved_count"] = saved_count

    state["running"] = False
    if preview_thread is not None:
        preview_thread.join(timeout=2.0)

    release()
    print(f"Done. Saved {saved_count} images to {output_dir}")
    return saved_count


def capture_manual_loop(read_frame, release, output_dir, camera_name, board_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    index = next_index(output_dir)
    window = f"calibration-{camera_name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok_frame, frame = read_frame()
        if not ok_frame or frame is None:
            print("Frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_board, corners = detect_board(gray, board_size)
        display = frame.copy()
        if ok_board:
            cv2.drawChessboardCorners(display, board_size, corners, ok_board)
        draw_status(display, ok_board, saved_count, camera_name, 1, saved_count, 0, manual=True)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and ok_board:
            path = output_dir / f"img_raw{index}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")
            saved_count += 1
            index += 1
            time.sleep(0.3)

    release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_count} images to {output_dir}")
    return saved_count


def capture_batch_loop(read_frame, release, output_dir, camera_name, board_size, batch_size, batch_interval, max_batches):
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    index = next_index(output_dir)
    batch_no = 1

    print(f"Batch mode: {batch_size} photos per batch, {batch_interval}s between photos in a batch.")
    print("Move the chessboard to a new position between batches.")

    while True:
        if max_batches > 0 and batch_no > max_batches:
            break

        print(f"\n=== Batch {batch_no}: capturing up to {batch_size} photos ===")
        batch_saved, index = capture_batch(
            read_frame,
            output_dir,
            camera_name,
            board_size,
            batch_size,
            batch_interval,
            batch_no,
            index,
        )
        total_saved += batch_saved
        print(f"Batch {batch_no} saved {batch_saved} photo(s). Total so far: {total_saved}")

        if max_batches > 0 and batch_no >= max_batches:
            break
        if not wait_for_next_batch(batch_no):
            break
        batch_no += 1

    release()
    print(f"Done. Saved {total_saved} images to {output_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Capture chessboard images from one camera.")
    parser.add_argument(
        "--camera",
        choices=["front", "left", "right", "rear"],
        default="left",
        help="Which camera to test first. Default: left USB.",
    )
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--board-cols", type=int, default=7, help="Inner corner count horizontally.")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner corner count vertically.")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--manual", action="store_true", help="Step mode: live preview + ENTER in terminal to save.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window in step mode.")
    parser.add_argument("--count", type=int, default=15, help="Step mode: stop after N saved photos (0 = until q).")
    parser.add_argument("--auto-count", type=int, default=0, help="Auto mode: save N photos when board is visible.")
    parser.add_argument("--auto-interval", type=float, default=3.0, help="Auto mode: seconds between saves.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch mode: photos per batch.")
    parser.add_argument("--batch-interval", type=float, default=3.0, help="Seconds between photos inside one batch.")
    parser.add_argument("--batches", type=int, default=0, help="Max batches (0 = until you press q at prompt).")
    parser.add_argument("--fresh", action="store_true", help="Delete existing img_raw*.jpg in output dir before capture.")
    args = parser.parse_args()

    hardware = load_hardware()
    camera = hardware[args.camera]
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT / args.camera
    board_size = (args.board_cols, args.board_rows)

    if args.fresh and output_dir.exists():
        for path in output_dir.glob("img_raw*.jpg"):
            path.unlink()
        print(f"Cleared old captures in {output_dir}")

    print(f"Camera: {args.camera} ({camera['model']})")
    print(f"Calibration mode: {camera['calibration_mode']}")
    print(f"Output: {output_dir}")

    if camera["interface"] == "usb":
        device_index, resolved_from = resolve_usb_capture_index(camera)
        print(f"USB device: /dev/video{device_index} ({resolved_from})")
        capture = open_usb(device_index, args.width, args.height, args.warmup, args.warmup_frames)

        def read_frame():
            return capture.read()

        def release():
            capture.release()

    elif camera["interface"] == "csi":
        camera_num = 1 if args.camera == "front" else 0
        csi = open_csi(camera_num, args.width, args.height, args.warmup, args.warmup_frames)

        def read_frame():
            return read_csi(csi)

        def release():
            csi.close()

    else:
        raise RuntimeError(f"Unsupported interface: {camera['interface']}")

    if args.manual:
        count = capture_step_loop(
            read_frame,
            release,
            output_dir,
            args.camera,
            board_size,
            preview=not args.no_preview,
            max_count=args.count,
        )
    elif args.auto_count > 0:
        count = capture_auto_loop(
            read_frame,
            release,
            output_dir,
            args.camera,
            board_size,
            args.auto_count,
            args.auto_interval,
        )
    else:
        count = capture_batch_loop(
            read_frame,
            release,
            output_dir,
            args.camera,
            board_size,
            args.batch_size,
            args.batch_interval,
            args.batches,
        )

    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
