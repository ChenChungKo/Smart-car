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

超音波與 SG90 雲台同座，確認距離前先鎖在已儲存的正前方（`gimbal_home.json`）：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 check_ultrasonic.py
```

用卷尺從**探頭金屬面**量到正前方平面，對照螢幕上的 `median=`。建議依序放在 15、20、30、50 cm。若已知卷尺距離：

```bash
python3 check_ultrasonic.py --expect 20
```

誤差約 ±3 cm 內即可用於巡航。`Ctrl+C` 結束。

舊的短測仍可用：

```bash
python3 full_car_test.py --test ultrasonic --samples 20
```

## 車頭 CSI 距離辨識

巡航用車頭 CSI（Picamera2 `camera_num=1`）的是左／中／右**堵塞分數**，不是直接輸出公分。這個程式會抓牆／地面交界估一個 `vis=` 公分，並用超音波 `sonic=` 當對照：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 check_front_csi_distance.py
```

把平面放在車頭正前方 15、20、30、50 cm。紅線應落在**牆腳**（泡棉底與地板交界），不是泡棉上緣。若 `vis=` 仍偏，對準後按 `c` 用超音波校正俯角（寫入 `csi_front_geom.json`）。`s` 存圖到 `check_front_csi_debug/`，`q` 結束。

巡航轉向看的是 `score M`：約 0.36 開始避障，不是 `vis` 公分。

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

先把雲台設成正前方（程式一啟動就轉過去並維持）。可趁伺服出力時把雲台螺絲對準車頭再鎖緊，也可用按鍵微調後按 Enter 儲存：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 setup_gimbal_servo.py
```

儲存檔為 `gimbal_home.json`。之後巡航一開始就讀這個角度，不會等相機暖機才轉正。

斷電後 SG90 沒有 PWM 會被重力拉歪，這是正常的。以前每次建立 `Servo()` 還會先把脈衝設成 1500μs（約 80°）再回到你存的角度，所以重開後看起來會偏掉。現在會直接回到 `pan=76 / tilt=90`。上電後請跑一次設定或巡航，雲台才會再出力：

```bash
python3 setup_gimbal_servo.py
```

若仍偏幾度，趁伺服出力時把雲台齒盤重新對準車頭再鎖緊；便宜齒輪回程差約 2–5°。

掃描測試：

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

## 安全環景巡航 MVP

程式：

```text
Code/Server/safe_surround_cruise.py
```

相機角色：

```text
front = CSI Picamera2 camera_num=1（控制）
left  = USB，依 camera_hardware.json 的 usb_bus 動態解析（控制）
right = USB，依 camera_hardware.json 的 usb_bus 動態解析（控制）
rear  = CSI Picamera2 camera_num=0（固定後方控制）
gimbal = USB bus usb-xhci-hcd.0-1.2（行駛時 SG90 鎖定 90/90）
```

USB 的 `/dev/videoX` 可能在重開機後改變，不要把編號寫死。程式會用
`v4l2-ctl --list-devices` 的 bus 資訊解析。相機與超音波共用雲台，因此**行駛
時雲台永遠固定在 pan/tilt 90/90**，不可邊開邊轉。若把超音波轉到側面，側向
空隙（例如 180 cm）會被當成前方暢通，車就會在牆壁前衝出去或卡卡地抽動。
控制用的距離永遠是最近一次「朝前且已穩定」的測距；側向讀數不參與避障。
啟動時雲台回中後等待 `--gimbal-settle-s`（預設 0.40 秒）才開始用超音波。
`Ctrl+C` 或正常結束時會回到 90/90；診斷時可用 `--no-gimbal`。

### 第一次執行：只看決策、不動馬達

先關閉全車測試 GUI、其他相機預覽與巡航程式，再執行：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 safe_surround_cruise.py --max-frames 100
```

沒有 `--arm` 時程式不會建立馬達物件。日誌會顯示：

```text
risk F/L/R/B=...
age=...s
sonic=...cm
ir=...
action=...
pwm=(...)
```

即使 `pwm` 顯示非零，dry-run 也只代表「預計命令」，輪子不會動。

### 架高輪子後啟用馬達

先確認：

- 車板電源開關已開
- 電池至少 7.0V
- `front`、`left`、`right`、`rear` 相機都顯示 `ok`
- 後方留有空間
- 人員可立即按 `Ctrl+C`

再用低速短測：

```bash
python3 safe_surround_cruise.py \
  --arm \
  --speed 900 \
  --turn-speed 900 \
  --back-speed 900 \
  --max-frames 100
```

只有明確提供 `--arm` 且 preflight 通過才會初始化馬達。任何必要相機過期、
控制迴圈超時都會輸出四輪 PWM=0。`Ctrl+C`、SIGTERM 和正常離開也都會在
`finally` 再停一次並讓雲台回中。

車板沒有獨立電源回報 GPIO；IR 與循線感測器即使有供電也可能全部輸出 LOW。
因此 `board=unknown(high_pins=0)` 不代表開關關閉，預設只警告並由操作者確認。
若環境能保證至少一個感測 GPIO 在通電時為 HIGH，可加 `--require-board-sense`
恢復嚴格拒絕模式。

目前 1:120 馬達在約 500 PWM 時無法克服靜摩擦。程式預設
`--motor-min-pwm 900`：若 `--speed`、`--turn-speed` 或 `--back-speed`
低於此值，真正送給主要驅動輪的 PWM 仍會提高到 900，並在日誌中顯示實際值。
更換馬達或確認較低值可以起步後，才降低此參數。

預設超音波在 18 cm 開始轉向；以 PWM 900 的慣性，目標是最接近約 15 cm，
但地面摩擦、速度與超音波角度都會影響實際距離。側面影像風險需達 0.70
才會主動偏離，降至 0.45 才解除，避免啟動時因單張側面影像誤判而立即彎行。
`slow` 會交替輸出移動 PWM 與零 PWM，而不是送出無法起步的低 PWM。
避障一旦選定左轉或右轉，至少保持 0.8 秒，並持續到超音波距離大於
30 cm 才解除；這可防止左右風險接近時來回切換方向。
前方 IR 會禁止繼續直行並觸發避障轉向，左右 IR 會禁止往該側避讓。中央前方
IR 只有在超音波進入 18 cm、或超音波暫時沒讀到時才參與決策，避免在約
40 cm 就提早轉向。左前／右前 IR 的距離閘門較寬（28 cm），斜向靠近牆角時
先轉車頭。超音波低於 32 cm 時即使畫面看起來空曠也改為慢行。

超音波低於 12 cm 時優先原地轉向離開牆壁；左右都堵住且後方畫面清楚才倒車。
後方相機畫面清楚時，即使後方 IR 亮著也視為後方可走（該 IR 常會誤觸）。
左右與後方都不安全才會 `hard_stop_boxed`。距離恢復至 25 cm 後才解除倒車。

雲台 USB 斷線只重開雲台相機，不會連帶關閉左／右 USB，也**不會去搶**左側
已在使用的 `/dev/video0`。左側或右側畫面過期
時仍可用車頭 CSI 與超音波繼續避障；只有車頭 CSI 過期且超音波也不再靠近
時才會 `fault_camera_stale`。USB 節點仍依 `usb_bus` 重新解析。
若 `v4l2-ctl --list-devices` 完全沒有 USB Camera，需重新插接相機／USB hub
或重新開機；程式不會在必要相機缺失時繼續行駛。

### 落地測試

架高測試正常後才移到寬敞地面：

```bash
python3 safe_surround_cruise.py --arm --speed 900 --turn-speed 900
```

控制器使用固定前／左／右／後四向風險。USB37 雲台與超音波共用支架，行駛時
不轉向，避免側向空隙被當成前方暢通。第一版不在每個控制週期拼接 1000×1000 BEV。牆面、透明物、
極低障礙與逆光仍可能造成視覺誤判，因此超音波與 IR 不應關閉；
`--no-ir`、`--no-sonic` 只供診斷。

若要儲存車頭疊圖：

```bash
python3 safe_surround_cruise.py --max-frames 100 --debug-every 10
```

輸出位於 `Code/Server/safe_surround_debug/`。

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
