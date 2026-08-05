"""Resolve USB camera /dev/video nodes (indexes change after reboot)."""

import re
import subprocess
from pathlib import Path

# IMX219 native 640x480 mode center-crops the sensor (~1280x960 region) and looks zoomed in.
# Scale from this full 4:3 mode instead when output is smaller.
IMX219_FULL_FOV_RAW = (1640, 1232)


def create_csi_still_configuration(camera, width, height):
    """Picamera2 still config that keeps IMX219 full field of view when downscaling."""
    full_w, full_h = IMX219_FULL_FOV_RAW
    if width <= full_w and height <= full_h:
        target_aspect = width / height
        full_aspect = full_w / full_h
        if abs(target_aspect - full_aspect) < 0.05:
            return camera.create_still_configuration(
                main={"size": (width, height)},
                raw={"size": IMX219_FULL_FOV_RAW},
            )
    return camera.create_still_configuration(main={"size": (width, height)})


def bev_size_scale(camera_name, default_scale=2.0):
    """All four cameras are now calibrated with the fisheye model; same scale for all.

    (Historically front used a "normal"/pinhole model whose 2x undistort cropped to a
    circle, so this forced 1.0 for front only. That model was wrong for this lens —
    see camera_hardware.json front notes — so front no longer needs the override.)
    """
    return default_scale


def bev_focal_scale(camera_name, default_focal_scale=1.0):
    """Per-camera focal-scale override hook (currently a no-op passthrough).

    Zooming the front CSI camera in (focal_scale > 1) sharpens/straightens the
    nearby chessboard but shrinks the mapped ground footprint, leaving black
    gaps at the front sector's far corners in the stitched BEV — a net loss.
    Kept as an explicit hook in case a future tuning finds a value that helps
    without that coverage trade-off.
    """
    return default_focal_scale


def parse_video_index(device_text):
    if "/dev/video" in device_text:
        return int(device_text.rsplit("video", 1)[1])
    raise ValueError(f"Unsupported USB device string: {device_text}")


def list_usb_capture_nodes():
    """Return {usb_bus: first_capture_video_index} from v4l2-ctl."""
    try:
        text = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("v4l2-ctl --list-devices failed") from exc

    mapping = {}
    current_bus = None
    for line in text.splitlines():
        bus_match = re.search(r"\((usb-[^)]+)\)", line)
        if bus_match and "USB" in line:
            current_bus = bus_match.group(1)
            continue
        dev_match = re.match(r"\t/dev/video(\d+)", line)
        if dev_match and current_bus and current_bus not in mapping:
            mapping[current_bus] = int(dev_match.group(1))
    return mapping


def resolve_usb_capture_index(camera_entry):
    """
    Pick the capture node for one USB camera entry from camera_hardware.json.
    Prefer stable usb_bus; fall back to configured device if it opens.
    """
    import cv2

    usb_bus = camera_entry.get("usb_bus", "")
    configured = camera_entry.get("device", "")

    if usb_bus:
        nodes = list_usb_capture_nodes()
        if usb_bus in nodes:
            return nodes[usb_bus], f"usb_bus {usb_bus} -> /dev/video{nodes[usb_bus]}"

    if configured:
        index = parse_video_index(configured)
        probe = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if probe.isOpened():
            probe.release()
            return index, f"configured {configured}"
        probe.release()

    if usb_bus:
        nodes = list_usb_capture_nodes()
        available = ", ".join(f"{bus}=/dev/video{idx}" for bus, idx in sorted(nodes.items()))
        raise RuntimeError(
            f"Cannot open USB camera (bus={usb_bus}, device={configured}). "
            f"Available USB capture nodes: {available or 'none'}"
        )

    raise RuntimeError(f"Cannot open USB camera device {configured}")
