## Smart Car

這個 repository 是基於 Freenove 4WD Smart Car Kit for Raspberry Pi 改裝後的個人測試版本。  
目前重點是保存整台車的功能測試、續航測試與硬體校正參數，方便日後重新部署或維修時快速確認。

詳細操作筆記請看：

```text
CAR_TESTING_GUIDE.md
```

## 目前車體改裝摘要

- 馬達齒比改為 `1:120`，原本為 `1:48`
- 2 個 Pi Camera / CSI camera
- 3 個 USB camera
- 車頭 2 個 servo
- 底部 3 個循跡感測器
- 額外 6 個紅外線避障感測器
- 外接 USB 麥克風與喇叭
- 電池電壓偵測與百分比估算
- LED 目前先跳過測試

## 主要測試程式

進入 Server 目錄：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
```

整車系統測試：

```bash
python3 full_car_test.py --test system
```

只測試五顆相機：

```bash
python3 full_car_test.py --test cameras
```

只測試喇叭、麥克風錄音與錄音回放：

```bash
python3 full_car_test.py --test audio
```

續航壓力測試：

```bash
python3 endurance_test.py
```

## 整車系統測試內容

`full_car_test.py --test system` 會依序測：

- 電池電壓與光敏 ADC
- 超聲波感測器
- 底部車道線循跡感測器
- 額外 6 個紅外線避障感測器
- 車頭 2 個 servo
- 蜂鳴器
- 外接喇叭播放、麥克風錄音與錄音回放
- 5 個相機（2 CSI + 3 USB，依序開啟）
- 基本馬達移動
- 麥克納姆輪移動

LED 預設跳過。若之後要包含 LED：

```bash
python3 full_car_test.py --test system --include-led
```

## 已校正參數

馬達基本測試：

```bash
python3 full_car_test.py --test basic-drive
```

目前預設校正值：

```text
turn-speed = 900
left-turn-90-duration = 2.4
right-turn-90-duration = 2.1
```

額外 6 個紅外線避障感測器 GPIO：

```text
GPIO26, GPIO20, GPIO19, GPIO16, GPIO6, GPIO12
```

避障感測器目前為 LOW 觸發：

```text
GPIO 讀到 LOW  -> obstacle=1
GPIO 讀到 HIGH -> obstacle=0
```

USB 相機配置：

```text
front = CSI Picamera camera_num=1
left  = usb-xhci-hcd.1-1
right = usb-xhci-hcd.0-2
rear  = usb-xhci-hcd.0-1.2
```

USB 的 `/dev/videoX` 編號可能在重開機或重新插拔後改變。程式會根據
`camera_hardware.json` 的 `usb_bus`，在每次拍攝前重新尋找該相機的第一個
V4L2 capture node，不應只依賴固定的 `/dev/videoX`。

USB 音效卡：

```text
plughw:2,0
```

音效卡編號也可能在重開機後改變，請用 `aplay -l` 與 `arecord -l` 確認。

## 五相機拍攝設定

目前 `python3 full_car_test.py --test cameras` 使用：

```text
CSI camera 0、1 : 1920x1440（4:3，保留 IMX219 完整視角）
USB left/right/rear: 1920x1080，YUYV，6 FPS
```

三顆 USB 相機皆為 `BL 1080p S10`，使用相同拍攝設定。YUYV 可避免部分 USB
鏈路在 1920x1080 MJPEG 模式發生 `Corrupt JPEG data`。USB 測試照片會寫入
相機位置、型號、感光元件、拍攝時間與軟體等 EXIF 資訊。

輸出位置：

```text
Code/Server/camera_test_images/
```

## 相機標定與 BEV

目前四向環景配置：

```text
front = CSI camera_num=1（pinhole）
left/right/rear = USB fisheye
```

主要工具：

```text
Code/Server/calibration_patterns/calibration_capture.py
Code/Server/calibration_patterns/calibration_intrinsic.py
Code/Server/capture_labeled_preview.py
Code/Server/plane_map_calibrate.py
Code/Server/plane_map_stitch.py
Code/Server/bev_extrinsic.py
Code/Server/bev_extrinsic_chessboard.py
Code/Server/bev_deploy.py
Code/Server/bev_stitch.py
```

相機硬體對應、內參 K/D、外參 H、棋盤照片及目前 BEV 除錯輸出皆保存在
repository，方便重建目前的標定狀態。

## 續航壓力測試

一般續航測試：

```bash
python3 endurance_test.py
```

車子架高後，啟用馬達負載：

```bash
python3 endurance_test.py --enable-motor-load --yes
```

先跑 10 分鐘確認流程：

```bash
python3 endurance_test.py --enable-motor-load --yes --max-minutes 10
```

停止條件：

```text
連續 3 次電壓 <= 6.8V 才停止
```

續航紀錄會輸出到：

```text
Code/Server/endurance_logs/
```

## 重開機後需要確認

重開機後通常可以直接執行：

```bash
python3 full_car_test.py --test system
```

若音訊或相機失效，先確認 USB 編號是否改變。

確認 USB 音效卡：

```bash
aplay -l
arecord -l
```

確認 USB 相機：

```bash
v4l2-ctl --list-devices
```

目前音訊設定：

```text
USB audio: plughw:2,0
```

USB 相機請以 `v4l2-ctl --list-devices` 重新確認；程式會用 `usb_bus` 自動解析。

## 重要檔案

```text
CAR_TESTING_GUIDE.md
Code/Server/full_car_test.py
Code/Server/endurance_test.py
Code/Server/params.json
Code/Server/camera_hardware.json
Code/Server/camera_devices.py
Code/Server/plane_map_calibration.json
Code/Server/calibration_patterns/
```

## 原始專案來源

本專案基於 Freenove 4WD Smart Car Kit for Raspberry Pi 修改。原始教學、PDF、圖片與範例程式仍保留在 repository 內，方便查閱硬體接線與官方說明。