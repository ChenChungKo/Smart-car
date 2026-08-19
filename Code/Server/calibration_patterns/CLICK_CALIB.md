# Click-Calib on Smart Car

Official upstream (WoodScape radial_poly + 6DoF): `click_calib_upstream/`

Smart-car adapter (OpenCV K/D + BEV H): `../click_calib_smartcar.py`

## Why an adapter?

Click-Calib expects WoodScape-style fisheye params (`k1..k4`, quaternion pose).  
This project uses OpenCV `K/D` + homography `H`. The adapter keeps the same idea:

1. Click corresponding ground points in adjacent overlaps  
2. Minimize BEV distance between those points  
3. Export refined `H` for `bev_deploy.py` / `bev_stitch.py`

## Steps

```bash
cd Code/Server

# 0) Need a decent initial H (chessboard extrinsic)
#    python3 bev_extrinsic_chessboard.py ...

# 1) Click pairs: front-left, front-right, rear-left, rear-right
#    Prefer green-mat grid intersections visible in BOTH cameras (>=8 each pair)
python3 click_calib_smartcar.py --click

# 2) Optimize
python3 click_calib_smartcar.py --optimize

# 3) Deploy + stitch
python3 bev_deploy.py --extrinsic-dir calibration_patterns/bev_extrinsic_click
python3 bev_stitch.py --auto-car-size
```

Optional: run official sample optimizer on bundled WoodScape keypoints:

```bash
python3 click_calib_smartcar.py --demo-upstream
```

## Click UI keys

- Left image then right image alternately (same index = same world point)
- `Enter` / `Space` = save pair
- `u` = undo, `r` = reset, `q` = skip pair
