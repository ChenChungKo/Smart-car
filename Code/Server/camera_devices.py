"""Resolve USB camera /dev/video nodes (indexes change after reboot)."""

import re
import subprocess
import threading
import time
from pathlib import Path

_USB_NODES_LOCK = threading.Lock()
_USB_NODES_CACHE: tuple[float, dict] = (0.0, {})

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
                main={"size": (width, height), "format": "RGB888"},
                raw={"size": IMX219_FULL_FOV_RAW},
            )
    return camera.create_still_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )


def create_csi_preview_configuration(camera, width=640, height=480):
    """Same pipeline as single-shot CSI captures (camera_test_images/picamera_*.jpg).

    Video mode is a different colour path; live preview must use still config.
    """
    return create_csi_still_configuration(camera, width, height)


def csi_array_to_bgr(arr):
    """Picamera2 capture_array() on this Pi is already OpenCV BGR.

    Format is advertised as RGB888, but the numpy buffer matches the JPEG from
    capture_file() only if treated as BGR. RGB2BGR here turns purple into orange.
    """
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].copy()
    return arr


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


def list_usb_capture_nodes(*, force: bool = False):
    """Return {usb_bus: first_capture_video_index} from v4l2-ctl."""
    global _USB_NODES_CACHE
    now = time.monotonic()
    with _USB_NODES_LOCK:
        cached_at, cached = _USB_NODES_CACHE
        if not force and cached and now - cached_at < 1.0:
            return dict(cached)
    try:
        text = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("v4l2-ctl --list-devices failed") from exc

    mapping = {}
    current_bus = None
    for line in text.splitlines():
        if not line.startswith("\t") and line.strip():
            bus_match = re.search(r"\((usb-[^)]+)\)", line)
            current_bus = (
                bus_match.group(1) if bus_match and "USB" in line else None
            )
            continue
        dev_match = re.match(r"\t/dev/video(\d+)", line)
        if dev_match and current_bus and current_bus not in mapping:
            mapping[current_bus] = int(dev_match.group(1))
    with _USB_NODES_LOCK:
        _USB_NODES_CACHE = (time.monotonic(), mapping)
    return dict(mapping)


def resolve_usb_capture_index(camera_entry, occupied=None, force_list=False):
    """
    Pick the capture node for one USB camera entry from camera_hardware.json.
    Prefer stable usb_bus. Never probe-open a node (that resets other UVC
    cameras) and never return a node already claimed by another role.
    """
    occupied = {int(idx) for idx in (occupied or [])}
    usb_bus = camera_entry.get("usb_bus", "")
    configured = camera_entry.get("device", "")
    try:
        nodes = list_usb_capture_nodes(force=force_list)
    except RuntimeError:
        nodes = {}

    if usb_bus and usb_bus in nodes:
        index = nodes[usb_bus]
        if index in occupied:
            raise RuntimeError(
                f"USB camera bus={usb_bus} mapped to /dev/video{index} "
                "already used by another camera; not stealing it"
            )
        return index, f"usb_bus {usb_bus} -> /dev/video{index}"

    if configured:
        index = parse_video_index(configured)
        owner_bus = next(
            (bus for bus, idx in nodes.items() if idx == index),
            None,
        )
        foreign = bool(usb_bus and owner_bus and owner_bus != usb_bus)
        if index not in occupied and not foreign:
            return index, f"configured {configured}"

    if usb_bus:
        available = ", ".join(
            f"{bus}=/dev/video{idx}" for bus, idx in sorted(nodes.items())
        )
        raise RuntimeError(
            f"Cannot open USB camera (bus={usb_bus}, device={configured}). "
            f"Available USB capture nodes: {available or 'none'}"
        )

    raise RuntimeError(f"Cannot open USB camera device {configured}")
