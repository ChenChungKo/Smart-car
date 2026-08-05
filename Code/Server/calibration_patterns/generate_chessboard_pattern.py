#!/usr/bin/env python3
"""Generate OpenCV-compatible chessboard calibration patterns for printing."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_checkerboard_image(squares_x, squares_y, square_px, margin_px):
    width = squares_x * square_px + margin_px * 2
    height = squares_y * square_px + margin_px * 2
    image = np.full((height, width), 255, dtype=np.uint8)
    origin_x = margin_px
    origin_y = margin_px
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x1 = origin_x + col * square_px
                y1 = origin_y + row * square_px
                image[y1 : y1 + square_px, x1 : x1 + square_px] = 0
    return image


def build_svg(squares_x, squares_y, square_mm, page_width_mm, page_height_mm):
    pattern_width = squares_x * square_mm
    pattern_height = squares_y * square_mm
    origin_x = (page_width_mm - pattern_width) / 2.0
    origin_y = (page_height_mm - pattern_height) / 2.0
    rects = []
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x = origin_x + col * square_mm
                y = origin_y + row * square_mm
                rects.append(
                    f'  <rect x="{x:.3f}" y="{y:.3f}" width="{square_mm:.3f}" '
                    f'height="{square_mm:.3f}" fill="black" stroke="none"/>'
                )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width_mm}mm" '
        f'height="{page_height_mm}mm" viewBox="0 0 {page_width_mm} {page_height_mm}">',
        f'  <rect x="0" y="0" width="{page_width_mm}" height="{page_height_mm}" fill="white"/>',
        *rects,
        f'  <text x="5" y="{page_height_mm - 5}" font-size="3" fill="#444">'
        f'OpenCV chessboard: inner corners {squares_x - 1}x{squares_y - 1}, '
        f'square={square_mm}mm, page={page_width_mm}x{page_height_mm}mm</text>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate OpenCV chessboard calibration pattern.")
    parser.add_argument("--inner-cols", type=int, default=7, help="Inner corner count horizontally.")
    parser.add_argument("--inner-rows", type=int, default=6, help="Inner corner count vertically.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Square edge length in millimeters.")
    parser.add_argument("--page-width-mm", type=float, default=210.0, help="Page width, default A4.")
    parser.add_argument("--page-height-mm", type=float, default=297.0, help="Page height, default A4.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG print resolution.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for generated pattern files.",
    )
    args = parser.parse_args()

    squares_x = args.inner_cols + 1
    squares_y = args.inner_rows + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    square_px = int(round(args.square_mm / 25.4 * args.dpi))
    margin_px = int(round(10.0 / 25.4 * args.dpi))
    png = build_checkerboard_image(squares_x, squares_y, square_px, margin_px)
    png_path = output_dir / f"chessboard_{args.inner_cols}x{args.inner_rows}_{int(args.square_mm)}mm_{args.dpi}dpi.png"
    svg_path = output_dir / f"chessboard_{args.inner_cols}x{args.inner_rows}_{int(args.square_mm)}mm_A4.svg"
    readme_path = output_dir / "PRINTING.txt"

    cv2.imwrite(str(png_path), png)
    svg_path.write_text(
        build_svg(squares_x, squares_y, args.square_mm, args.page_width_mm, args.page_height_mm),
        encoding="utf-8",
    )

    pattern_width_mm = squares_x * args.square_mm
    pattern_height_mm = squares_y * args.square_mm
    readme_path.write_text(
        "\n".join(
            [
                "OpenCV chessboard calibration pattern",
                "=================================",
                f"Inner corners: {args.inner_cols} x {args.inner_rows}",
                f"Square size: {args.square_mm} mm",
                f"Pattern size: {pattern_width_mm} x {pattern_height_mm} mm",
                f"Page: A4 ({args.page_width_mm} x {args.page_height_mm} mm)",
                "",
                "CameraCalibration parameters:",
                f"  -bw {args.inner_cols} -bh {args.inner_rows} -size {int(args.square_mm)}",
                "",
                "Print instructions:",
                "1. Prefer chessboard_7x6_25mm_A4.svg in a browser or vector tool.",
                "2. Print at 100% scale. Do NOT use 'fit to page'.",
                "3. After printing, measure one square with a ruler.",
                "4. Paste the board onto flat cardboard/acrylic.",
                "",
                f"PNG file: {png_path.name}",
                f"  Image DPI metadata target: {args.dpi}",
                f"  Expected square size when printed correctly: {args.square_mm} mm",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved notes: {readme_path}")
    print(f"Inner corners: {args.inner_cols}x{args.inner_rows}, square={args.square_mm}mm")


if __name__ == "__main__":
    main()
