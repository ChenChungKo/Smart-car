# 四相機環景：啟動、拍攝與注意事項

更新日期：2026-08-06  
外參目錄：`calibration_patterns/bev_extrinsic_metric_auto/`（公制棋盤，**無俯看圖**）

---

## 1. 相機對應（勿憑 video 編號硬記）

| 位置 | 介面 | 解析方式 | 備註 |
|------|------|----------|------|
| front | CSI Picamera2 | `camera_num=1` | IMX219，用 `create_csi_still_configuration` 保留全 FOV |
| left | USB | `camera_hardware.json` → `usb_bus: usb-xhci-hcd.1-1` | 編號會變；重開機後用 `v4l2-ctl --list-devices` 對 bus |
| right | USB | `usb_bus: usb-xhci-hcd.0-2` | 同上 |
| rear | USB | `usb_bus: usb-xhci-hcd.0-1.2` | 同上 |

每個 USB Camera 條目下**第一個** `/dev/videoX` 才是擷取節點（下一個常是 metadata）。

重開機後請先：

```bash
v4l2-ctl --list-devices
```

再用 `camera_devices.resolve_usb_capture_index()`（依 `usb_bus`），不要寫死 video0/10/37。

---

## 2. 建議拍攝順序與參數

**順序：先三路 USB，再 CSI front。**  
（先開 Picamera2 時 libcamera 也會列到 USB，易與 OpenCV V4L2 搶裝置／拖慢。）

### USB（left / right / rear）

| 項目 | 建議 |
|------|------|
| 格式 | **優先 YUYV**，失敗再 MJPG |
| 解析度 | 640×480 |
| 暖機 | 開啟後 sleep **≥ 2.5 s**，再丟棄約 25–40 幀 |
| 緩衝 | `CAP_PROP_BUFFERSIZE = 1` |
| 左側特規 | 連拍多幀，用撕裂分數挑最好的一張（見下） |

### CSI front

| 項目 | 建議 |
|------|------|
| API | Picamera2 + `create_csi_still_configuration(cam, 640, 480)` |
| 控制 | `AeEnable=True`, `AwbEnable=True` |
| 暖機 | ≥ 1.5–2 s，再 capture 約 15–20 幀後存檔 |

一鍵腳本：

```bash
cd Code/Server
python3 capture_live_surround.py
# 產出 camera_labeled_live_now/*.jpg
# 並可選 --stitch 直接拼環景
python3 capture_live_surround.py --stitch
```

---

## 3. 左側相機為何常失敗（已踩過的坑）

1. **MJPG 不完整影格**  
   - 症狀：畫面橫向錯位、底部一條「錯位帶」、地磚縫上下對不齊。  
   - 日誌常見：`Corrupt JPEG data: premature end of data segment`  
   - 對策：改 **YUYV**；多丟棄暖機幀；多抓候選幀挑撕裂最小者。

2. **暖機／曝光太短**  
   - 症狀：整張偏暗、噪點大。  
   - 對策：加長 sleep + 多讀幾幀再存。

3. **裝置被佔用／超時**  
   - 症狀：`select() timeout`、卡很久。  
   - 對策：先釋放 CSI；USB 一個一個開；不要同時用預覽佔住同一 video。

4. **鏡頭前被線材擋住**  
   - 症狀：畫面邊緣大片糊白／糊黑近物。  
   - 對策：整理左側線材，屬硬體遮擋，不是演算法。

---

## 4. 右側方向（非常重要，勿再改錯）

校正時棋盤角點方向有歧義。右側 `H` 必須套用（見 `metric_extrinsic_from_boards.py`）：

```text
right: flip_h + flip_v   （繞該路棋盤在 BEV 上的中心）
```

| 錯誤現象 | 意義 |
|----------|------|
| 綠膠帶在**右前**，實際在**右後** | 缺 `flip_v`（前後顛倒） |
| 左右物體鏡像錯 | 與 `flip_h` 有關 |

**2026-08-06 確認的正式版：**  
對「已含 flip_h」的 `camera_right_H.npy` 再以**右側棋盤 BEV 中心**做 `flip_v`（board_pivot），左右地縫差約 **dy≈16 px**。  
不要改用車身中心 pivot 覆蓋這一版，除非重新目視確認綠膠帶在右後且地縫仍齊。

驗證口訣：

1. 環景「上=車頭」。  
2. 右後輪附近應看到綠膠帶（若現場有放）。  
3. 左右地板橫縫應大致同一高度（dy 約十幾像素可接受）。

檔案：

- `calibration_patterns/bev_extrinsic_metric_auto/camera_right_H.npy`  
- 程式常數：`MANUAL_ORIENTATION_FIX["right"]`

---

## 5. 拍攝後拼環景

```bash
cd Code/Server
python3 bev_deploy.py \
  --extrinsic-dir calibration_patterns/bev_extrinsic_metric_auto \
  --labeled-dir camera_labeled_live_now

python3 bev_stitch.py --blend --feather 40 \
  --car-width 207 --car-height 333 \
  --car-center-x 506 --car-center-y 511 \
  --output-dir bev_output/live_now_stitch
```

車身遮罩數字來自  
`calibration_patterns/bev_extrinsic_metric_auto/metric_extrinsic_auto_summary.json` → `car_mask_px`。

---

## 6. 演算法一句話（給報告用）

幾何式環景（IPM）：魚眼去畸變 → 每路平面單應 `H` → 扇形遮罩＋羽化拼接。  
非深度學習 BEV。換場景可沿用同一組 `K/D/H`，前提是相機相對車身未動。
