"""Resolve USB camera /dev/video nodes (indexes change after reboot)."""

import re
import subprocess
from pathlib import Path


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
