#!/usr/bin/env python3
"""Print ArUco wall tags for route following.

Print the PDF at 100 % ("actual size", not "fit to page"), then measure the
black square with a ruler and write that side into arena_map.json tag_size_m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SERVER = Path(__file__).resolve().parent
OUT = SERVER / "aruco_print"
DPI = 300
A4_MM = (210.0, 297.0)
PAGE_MARGIN_MM = 10.0


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def tag_tile(dictionary, tag_id: int, size_mm: float, quiet_mm: float) -> np.ndarray:
    cells = dictionary.markerSize + 2
    side = max(cells, round(mm_to_px(size_mm) / cells) * cells)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side)
    quiet = mm_to_px(quiet_mm)
    label_h = mm_to_px(9.0)
    tile = np.full((side + 2 * quiet + label_h, side + 2 * quiet), 255, np.uint8)
    tile[quiet:quiet + side, quiet:quiet + side] = marker
    text = f"id {tag_id}   black square {side / DPI * 25.4:.1f} mm"
    cv2.putText(
        tile,
        text,
        (quiet, side + 2 * quiet + label_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        DPI / 300.0 * 1.1,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), 170, 1)
    return tile


def pack_pages(tiles: list[np.ndarray]) -> list[np.ndarray]:
    page_w, page_h = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])
    margin = mm_to_px(PAGE_MARGIN_MM)
    pages: list[np.ndarray] = []
    page = None
    x = y = row_h = 0
    for tile in tiles:
        th, tw = tile.shape[:2]
        if tw > page_w - 2 * margin or th > page_h - 2 * margin:
            raise SystemExit("Tag does not fit on A4; use a smaller --size-mm.")
        if page is None:
            page = np.full((page_h, page_w), 255, np.uint8)
            x = y = margin
            row_h = 0
        if x + tw > page_w - margin:
            x = margin
            y += row_h + margin // 2
            row_h = 0
        if y + th > page_h - margin:
            pages.append(page)
            page = np.full((page_h, page_w), 255, np.uint8)
            x = y = margin
            row_h = 0
        page[y:y + th, x:x + tw] = tile
        x += tw + margin // 2
        row_h = max(row_h, th)
    if page is not None:
        pages.append(page)
    return pages


def main() -> None:
    p = argparse.ArgumentParser(description="Generate printable ArUco tags (A4, 300 dpi).")
    p.add_argument("--ids", default="0-7", help="e.g. 0-7 or 0,1,5")
    p.add_argument("--size-mm", type=float, default=100.0, help="Black square side.")
    p.add_argument("--quiet-mm", type=float, default=12.0, help="White border around the tag.")
    p.add_argument("--dict", default="DICT_4X4_50")
    args = p.parse_args()

    ids: list[int] = []
    for part in args.ids.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-"))
            ids.extend(range(lo, hi + 1))
        elif part:
            ids.append(int(part))

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for tag_id in ids:
        tile = tag_tile(dictionary, tag_id, args.size_mm, args.quiet_mm)
        cv2.imwrite(str(OUT / f"tag_{tag_id:02d}.png"), tile)
        tiles.append(tile)

    pages = [Image.fromarray(page) for page in pack_pages(tiles)]
    pdf = OUT / f"aruco_{args.dict}_{args.size_mm:g}mm.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"{len(ids)} tags, {len(pages)} A4 page(s) -> {pdf}")
    print("Print at 100 % / actual size, then measure the black square.")


if __name__ == "__main__":
    main()
