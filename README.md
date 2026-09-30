## Smart Car

這個 repository 是基於 Freenove 4WD Smart Car Kit for Raspberry Pi 改裝後的個人測試版本。  
目前重點包含：

- 整車功能測試、續航測試、硬體校正參數
- 四相機幾何環景（IPM）拼接
- **場地內 ArUco 牆上標籤定位**，以及依航點停看走的路線跟隨

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

## 四相機環景（目前正式方法）

演算法是**幾何式環景 / IPM**（Inverse Perspective Mapping），不是深度學習 BEV。  
流程：魚眼去畸變 → 每路地面單應 `H` → 扇形遮罩＋羽化拼接。  
**不需要俯看圖。** 換場景可沿用同一組 `K/D/H`，前提是鏡頭相對車身沒被碰歪。

**拼接成功的關鍵：** 不是對齊俯拍照，而是把**四面棋盤在地面上的真實座標**（以及車身四角）量好寫進 layout。  
每路影像偵測棋盤角點後，與這些公制座標對應，才能估出把畫面拉到同一俯視平面的 `H`。四面座標缺一、或量錯，那路就會歪、對不齊。

座標檔：

```text
Code/Server/calibration_patterns/metric_layout_aug3.json
```

裡面包含：墊子尺寸、每路棋盤一個已知角的 (long, short) cm、車身四角 cm。  
腳本 `metric_extrinsic_from_boards.py` 依此算出 `bev_extrinsic_metric_auto/` 的四個 `H`。

四路對應（USB 編號會變，一律用 `usb_bus`）：

```text
front = CSI Picamera2 camera_num=1（IMX219，內參用 fisheye）
left  = USB usb-xhci-hcd.1-1
right = USB usb-xhci-hcd.0-2
rear  = USB usb-xhci-hcd.0-1.2
```

正式外參（公制棋盤）：

```text
Code/Server/calibration_patterns/bev_extrinsic_metric_auto/
```

右側 `H` 必須是 `flip_h + flip_v`（繞右側棋盤在 BEV 上的中心）。  
缺 `flip_v` 會把右後地面映到右前（例如綠膠帶會跑到右前輪）。  
2026-08-06 確認版地縫差約 dy≈16 px，不要改用車身中心當 pivot 覆蓋。

### 如何拍攝並拼出環景

1. 確認相機沒被其他程式佔用，重開機後先對一下 USB 節點：

```bash
v4l2-ctl --list-devices
```

每個 USB Camera 條目下**第一個** `/dev/videoX` 才是擷取節點。程式會依 `camera_hardware.json` 的 `usb_bus` 自動對應，不要寫死編號。

2. 進入 Server 目錄，四路各拍一張並立刻拼接：

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 capture_live_surround.py --stitch
```

腳本會：**先拍 USB left → right → rear，再拍 CSI front**（先開 CSI 容易跟 USB 搶裝置）。  
USB 優先用 YUYV；左側會連拍多幀，挑撕裂較小的一張。

3. 看結果：

```text
四路原圖：Code/Server/camera_labeled_live_now/{front,left,right,rear}.jpg
四路併圖：Code/Server/bev_output/live_now_stitch/raw_2x2.jpg
環景結果：Code/Server/bev_output/live_now_stitch/surround_square.jpg
```

只拍照、先不拼接：

```bash
python3 capture_live_surround.py
```

已有四張圖（檔名需含 `front` / `left` / `right` / `rear`）時，手動部署再拼接：

```bash
python3 bev_deploy.py \
  --extrinsic-dir calibration_patterns/bev_extrinsic_metric_auto \
  --labeled-dir camera_labeled_live_now

python3 bev_stitch.py --blend --feather 40 \
  --car-width 207 --car-height 333 \
  --car-center-x 506 --car-center-y 511 \
  --output-dir bev_output/live_now_stitch
```

車身遮罩數字來自  
`calibration_patterns/bev_extrinsic_metric_auto/metric_extrinsic_auto_summary.json` 的 `car_mask_px`。

常見狀況：

- 左側畫面像被橫切、地磚縫錯位：MJPG 撕裂。腳本已優先 YUYV。
- `select() timeout` 卡住很久：USB 被佔用或 MJPG 掛住。關掉其他預覽後重跑。
- 環景右後標物跑到右前：右側 `H` 缺 `flip_v`，不要自行覆蓋 `camera_right_H.npy`。
- 換場景拼得出圖，但對角黑洞／遠方模糊：幾何極限，不是沒拍到檔。

更細的注意事項：`Code/Server/BEV_LIVE_CAPTURE.md`

### 範例結果

左欄為四路原圖（`raw_2x2.jpg`），右欄為拼接環景（`surround_square.jpg`）。圖片在 `docs/bev_examples/`。

#### 在桌上

<table>
<tr>
<td width="50%"><img src="docs/bev_examples/在桌上/raw_2x2.jpg" alt="在桌上 四路原圖" /></td>
<td width="50%"><img src="docs/bev_examples/在桌上/surround_square.jpg" alt="在桌上 環景" /></td>
</tr>
<tr>
<td align="center">四路原圖</td>
<td align="center">環景結果</td>
</tr>
</table>

#### 在綠色桌墊

<table>
<tr>
<td width="50%"><img src="docs/bev_examples/在綠色桌墊/raw_2x2.jpg" alt="在綠色桌墊 四路原圖" /></td>
<td width="50%"><img src="docs/bev_examples/在綠色桌墊/surround_square.jpg" alt="在綠色桌墊 環景" /></td>
</tr>
<tr>
<td align="center">四路原圖</td>
<td align="center">環景結果</td>
</tr>
</table>

#### 放在地上

<table>
<tr>
<td width="50%"><img src="docs/bev_examples/放在地上/raw_2x2.jpg" alt="放在地上 四路原圖" /></td>
<td width="50%"><img src="docs/bev_examples/放在地上/surround_square.jpg" alt="放在地上 環景" /></td>
</tr>
<tr>
<td align="center">四路原圖</td>
<td align="center">環景結果</td>
</tr>
</table>

#### 有放磁鐵

<table>
<tr>
<td width="50%"><img src="docs/bev_examples/有放磁鐵/raw_2x2.jpg" alt="有放磁鐵 四路原圖" /></td>
<td width="50%"><img src="docs/bev_examples/有放磁鐵/surround_square.jpg" alt="有放磁鐵 環景" /></td>
</tr>
<tr>
<td align="center">四路原圖</td>
<td align="center">環景結果</td>
</tr>
</table>

主要程式：

```text
Code/Server/capture_live_surround.py      # 現場四路拍攝（可 --stitch）
Code/Server/metric_extrinsic_from_boards.py
Code/Server/bev_deploy.py
Code/Server/bev_stitch.py
Code/Server/camera_hardware.json
```

## 場地 ArUco 定位與路線跟隨

目標：在自建場地（目前內框約 `1.40 m × 1.40 m`）裡，用前後 CSI 魚眼鏡頭看牆上的 ArUco 標籤，算出車身中心的 `(x, y, yaw)`，再照航點停看走。

**定位用相機：** 只用前後兩顆 CSI（`front = camera_num=1`、`rear = camera_num=0`）。左右 USB 相機不參與定位。

**控制方式：** 不是邊走邊定位。`route_follow.py` 採用 stop-and-look：停下 → 等畫面穩定 → 用標籤定位 → 原地轉向 → 直走一小段（預設最多 0.35 m）→ 再停下。推車過程中即時視窗可能會抖，停下來後才是可靠結果。

### 座標與地圖

| 項目 | 說明 |
|---|---|
| 原點 | 左下角兩面牆**內側面**交點（左邊灰牆 × 下方藍牆） |
| x / y | x 沿下方藍牆往右，y 沿左邊灰牆往上，單位公尺 |
| 車身座標 | 原點在車身中心地面；x 朝車頭，y 朝車左，z 朝上 |
| 標籤字典 | `DICT_4X4_50`，黑色方塊邊長 **10 cm** |
| 標籤位置 | `Code/Server/arena_map.json`（目前 8 張，貼在 8 塊泡棉正中，中心離地 7.5 cm） |
| 鏡頭外參 | `Code/Server/car_nav.json` 的 `camera_mounts`（前後鏡頭離車心約 ±11 cm、離地約 4 cm） |

地圖裡的 `facing_deg` 是標籤正面朝場地內的方向；若標籤上下貼反，加 `"rotation_deg": 180`（目前 id 0、1）。標籤四角要用膠帶貼平，紙面翹起來會反光、解不出編號。

航點路線也寫在 `arena_map.json` 的 `routes`：

```text
loop  四角來回： (0.35,0.35) → (1.05,0.35) → (1.05,1.05) → (0.35,1.05) → 起點
line  橫向一線： (0.35,0.70) → (1.05,0.70)
```

之後放高台／斜坡／右下藍色方塊時，部分標籤會被擋，記得改貼位置並更新 `arena_map.json`。

### 印標籤

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 aruco_print.py --ids 0-7 --size-mm 100
```

輸出 PDF／PNG 在本機 `aruco_print/`（已列入 `.gitignore`，不進 Git）。列印後量黑色方塊是否剛好 10 cm；若不是，改 `arena_map.json` 的 `tag_size_m`。

### 即時定位（不動馬達）

```bash
cd /home/pi/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
python3 aruco_localizer.py
```

視窗左邊是前後鏡頭（有偵測到的標籤會畫框），右邊是場地地圖與軌跡。按 `s` 存原始畫面到 `aruco_debug/`，`q` 離開。

終端機會印類似：

```text
x=0.68 y=0.68 yaw=+0deg [f2,f3,r7,r6] rms 2.5px
```

`rms` 愈小愈好；超過約 6 px 會拒絕輸出，地圖停在上一次可靠位置（避免亂跳）。

常用參數：

```bash
# 只量鏡頭到標籤的距離（不需要地圖）
python3 aruco_localizer.py --range

# 車放平、看得清楚至少 2 張標籤時，粗估俯角（會寫回 car_nav.json）
# 注意：鏡頭高度/前後位置若不對，單點 --calibrate-pitch 會用錯誤俯角去湊，換位置就失敗。
# 目前 car_nav.json 的 pitch/z/x 已用多位置解過，平常不要再跑這個覆蓋掉。
python3 aruco_localizer.py --calibrate-pitch 20
```

定位演算法摘要：

1. 魚眼原圖 + 中央拉直後的圖各偵測一次，合併結果（近處彎曲標籤較易解出）。
2. 角點用 fisheye 內參還原成單位射線，多張標籤一起做平面 `(x, y, yaw)` 擬合。
3. 離光軸太遠、或側視太斜的標籤會被丟掉；只剩一張且超過約 0.8 m 時不採用（避免遠處單標籤亂跳）。
4. 保留彼此吻合最多的一組標籤，而不是只留「自己誤差最小」的那一張。

### 路線跟隨

預設是 dry-run（不轉馬達）。確認定位 OK 後再加 `--arm`。

```bash
# 先校正直走/轉彎速度（會動車，寫回 car_nav.json 的 motion）
python3 route_follow.py --calibrate-motion --arm

# 乾跑：只定位、印計畫，馬達不轉
python3 route_follow.py --route loop

# 實跑一圈
python3 route_follow.py --route loop --arm

# 反覆跑直到 Ctrl+C
python3 route_follow.py --route loop --arm --repeat
```

安全層（會覆寫導航指令）：

```text
超音波 < 18 cm     → 硬停
前紅外線           → 僅在超音波也夠近時才信
前鏡頭走廊分數     → 近障分數高且超音波落在 creep 範圍才信
電池電壓過低       → 中止
```

軌跡圖會寫到 `route_follow_debug/`。

### 建議驗證順序

1. `python3 aruco_print.py` → 貼標籤（四角貼平）→ 更新 `arena_map.json` 真實位置。
2. `python3 aruco_localizer.py --range`，在 12 cm / 26 cm 量距離是否準。
3. 車放場地中央，跑 `python3 aruco_localizer.py`，確認 `x≈0.70 y≈0.70`、`rms` 約 2–4 px。
4. 用手推到 2–3 個用尺量過的位置，比對畫面座標（誤差目標約 3 cm 內）。
5. `python3 route_follow.py --calibrate-motion --arm`。
6. `python3 route_follow.py --route loop` 乾跑，再 `--arm`。

### 相關檔案

```text
Code/Server/aruco_print.py          # 產生可列印標籤
Code/Server/aruco_localizer.py      # 即時定位 / 俯角校正 / 測距
Code/Server/route_follow.py         # 航點跟隨（stop-and-look）
Code/Server/arena_map.json          # 場地尺寸、標籤位姿、航點
Code/Server/car_nav.json            # 鏡頭外參、運動參數
Code/Server/safe_surround_cruise.py # 安全感測（route_follow 會用到）
Code/Server/vision_detector.py      # 前鏡頭走廊分數（route_follow 會用到）
```

除錯圖目錄（不進 Git）：`aruco_debug/`、`aruco_print/`、`route_follow_debug/`。

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
Code/Server/BEV_LIVE_CAPTURE.md
Code/Server/full_car_test.py
Code/Server/endurance_test.py
Code/Server/params.json
Code/Server/camera_hardware.json
Code/Server/camera_devices.py
Code/Server/capture_live_surround.py
Code/Server/calibration_patterns/bev_extrinsic_metric_auto/
Code/Server/aruco_localizer.py
Code/Server/route_follow.py
Code/Server/arena_map.json
Code/Server/car_nav.json
```

## 原始專案來源

本專案基於 Freenove 4WD Smart Car Kit for Raspberry Pi 修改。原始教學、PDF、圖片與範例程式仍保留在 repository 內，方便查閱硬體接線與官方說明。