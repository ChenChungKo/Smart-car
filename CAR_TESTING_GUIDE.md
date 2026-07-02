# Smart Car Testing Guide

這份文件記錄目前改裝後 Freenove 4WD Smart Car 的測試方式與已校正參數，避免之後忘記怎麼操作。

## 測試程式位置

主要測試程式都在：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
```

整車系統測試：

```bash
python3 full_car_test.py --test system
```

續航壓力測試：

```bash
python3 endurance_test.py
```

## 整車系統測試

執行：

```bash
python3 full_car_test.py --test system
```

系統測試會依序測：

- 電池電壓與光敏 ADC
- 超聲波感測器
- 底部車道線循跡感測器
- 額外 6 個紅外線避障感測器
- 車頭 2 個 servo
- 蜂鳴器
- 外接麥克風與喇叭
- 5 個相機
- 基本馬達移動
- 麥克納姆輪移動

LED 目前預設跳過。

如果之後要包含 LED 測試：

```bash
python3 full_car_test.py --test system --include-led
```

## 已校正馬達參數

目前馬達已改成 `1:120`，原本是 `1:48`。

基本移動測試：

```bash
python3 full_car_test.py --test basic-drive
```

目前預設校正值：

```text
turn-speed = 900
left-turn-90-duration = 2.4
right-turn-90-duration = 2.1
```

等同於：

```bash
python3 full_car_test.py --test basic-drive \
  --turn-speed 900 \
  --left-turn-90-duration 2.4 \
  --right-turn-90-duration 2.1
```

## 超聲波感測器

```bash
python3 full_car_test.py --test ultrasonic
```

增加樣本數：

```bash
python3 full_car_test.py --test ultrasonic --samples 20
```

## 底部車道線感測器

```bash
python3 full_car_test.py --test infrared
```

輸出中的 `combined` 是三個感測器的二進位合併：

```text
left middle right -> combined
0    0      0     -> 0
0    0      1     -> 1
0    1      0     -> 2
0    1      1     -> 3
1    0      0     -> 4
1    0      1     -> 5
1    1      0     -> 6
1    1      1     -> 7
```

## 額外 6 個紅外線避障感測器

目前 GPIO 腳位：

```text
GPIO26, GPIO20, GPIO19, GPIO16, GPIO6, GPIO12
```

測試：

```bash
python3 full_car_test.py --test obstacle
```

目前判斷是 LOW 觸發：

```text
GPIO 讀到 LOW  -> obstacle=1
GPIO 讀到 HIGH -> obstacle=0
```

## 車頭 Servo

```bash
python3 full_car_test.py --test servo
```

目前會測：

```text
channel 0: 0 -> 90 -> 180 -> 90
channel 1: 80 -> 90 -> 180 -> 90
```

如果要縮小測試角度：

```bash
python3 full_car_test.py --test servo \
  --servo0-min 75 --servo0-max 105 \
  --servo1-min 90 --servo1-max 120
```

## 相機測試

目前總共 5 個相機：

- 2 個 Pi Camera / CSI camera
- 3 個 USB camera

測試：

```bash
python3 full_car_test.py --test cameras
```

目前 USB camera 節點：

```text
/dev/video0
/dev/video2
/dev/video37
```

程式預設：

```text
--usb-camera-indexes 0,2,37
```

輸出照片會在：

```text
camera_test_images/
```

檔名：

```text
picamera_0.jpg
picamera_1.jpg
usb_camera_0.jpg
usb_camera_1.jpg
usb_camera_2.jpg
```

重開機後如果 USB 相機節點變了，用這個確認：

```bash
v4l2-ctl --list-devices
```

每個 `USB 2.0 Camera` 區塊的第一個 `/dev/videoX` 通常是 capture 節點。

如果節點變成 `0,2,4`，執行：

```bash
python3 full_car_test.py --test system --usb-camera-indexes 0,2,4
```

## 外接麥克風與喇叭

目前 USB 音效卡為：

```text
plughw:2,0
```

系統測試已預設使用：

```text
--speaker-device plughw:2,0
--mic-device plughw:2,0
```

單獨測音訊：

```bash
python3 full_car_test.py --test audio
```

如果聲音又從螢幕出來，先確認 USB 音效卡編號：

```bash
aplay -l
arecord -l
```

找這種輸出：

```text
card 2: Device [USB PnP Audio Device], device 0: USB Audio
```

代表使用：

```text
plughw:2,0
```

如果重開後變成 `card 3`，改用：

```bash
python3 full_car_test.py --test system \
  --speaker-device plughw:3,0 \
  --mic-device plughw:3,0
```

## 電池電量

測試：

```bash
python3 full_car_test.py --test adc
```

輸出會包含：

```text
battery=7.14V, battery_percent~=31%
```

百分比是 2S 鋰電池估算值，會受馬達負載與電池狀態影響。

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

續航測試會記錄 CSV：

```text
endurance_logs/endurance_YYYYMMDD_HHMMSS.csv
```

目前停止條件：

```text
連續 3 次電壓 <= 6.8V 才停止
```

測試其他截止電壓，例如 7.5V：

```bash
python3 endurance_test.py --enable-motor-load --yes --cutoff-voltage 7.5
```

如果中間電壓回升，低電壓計數會歸零。

## LED

LED 目前先跳過。

目前程式偵測到：

```text
Connect_Version=2
Pi_Version=2
driver=Freenove_SPI_LedPixel
SPI0-MOSI: GPIO10
```

若之後要排查 LED，優先確認：

- LED DIN 是否接到 `GPIO10 / SPI0 MOSI`
- LED 是否有供電
- LED GND 是否和 Raspberry Pi GND 共地
- LED 方向是否接到 `DIN` 而不是 `DOUT`
- 是否有其他裝置占用 `GPIO10`
- LED 是否為 WS2812/NeoPixel 相容

## 重開機後需要確認

通常直接執行即可：

```bash
python3 full_car_test.py --test system
```

重開機後最可能改變的是：

- USB 音效卡 card 編號
- USB 相機 `/dev/videoX` 節點

確認音效：

```bash
aplay -l
arecord -l
```

確認相機：

```bash
v4l2-ctl --list-devices
```

目前穩定值：

```text
USB audio: plughw:2,0
USB cameras: 0,2,37
```
