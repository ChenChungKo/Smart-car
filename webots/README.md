# Webots：用模擬場地跑 route_follow.py

車子的路線邏輯就是實車的 `Code/Server/route_follow.py`，Webots 只負責場地、感測器射線和 3D 畫面。

## 檔案

| 檔案 | 用途 |
|---|---|
| `protos/Freenove4WDSim.proto` | 車子（Z 軸朝上，R2025a）。尺寸與感測器位置照 `car_nav.json`：超聲波在雲台上、7 個紅外線、5 個鏡頭 |
| `controllers/route_follow_webots/` | 控制器：執行 `route_follow.py`，硬體換成 Webots 裝置 |
| `worlds/arena_lap.wbt` | 繞圈測試：每面牆中間一塊方塊，車子在原點 |
| `worlds/arena_course.wbt` | 另一張場地（平台、斜坡、叉形、風車），尚未放車 |
| `protos/Freenove4WD.proto`、`controllers/surround_cruise/` | 舊版（Y 軸朝上），保留不用 |

## 執行

1. 在電腦上安裝 Webots R2025a（Webots 不支援 Raspberry Pi）。
2. 安裝 Webots 使用的 Python 套件：`pip install numpy opencv-contrib-python`。
3. 開啟 `worlds/arena_lap.wbt`，按執行。控制台會印出跟實車一樣的 log，最後是 `WEBOTS RUN: finished` 或停下的原因。
4. 軌跡圖：`Code/Server/route_follow_webots_debug/trajectory.png`。

## 車子怎麼動

- **運動學模式**：車子沒有物理引擎，控制器每一步依照 `car_nav.json` 的速度（再乘上一點誤差）直接設定車子位置，所以麥克納姆輪的橫移不需要模擬滾子。
- **感測器**：超聲波、紅外線是 Webots 真的射線，會打到你場地裡的牆和方塊；超聲波用 sonar 類型，斜面會把聲波彈開，跟實車一樣。
- **定位**：用真實位置加上雜訊（約 1 公分、1 度），還沒有用鏡頭看標籤。
- **碰撞**：場景裡每個有 `boundingObject`（Box 或 IndexedFaceSet）的 Solid 都算障礙物，車身一碰到就停下並回報。

## 把車子放進其他世界

在 `.wbt` 開頭加：

```
EXTERNPROTO "../protos/Freenove4WDSim.proto"
```

最後加：

```
Freenove4WDSim {
  translation -0.515 -0.56 0
}
```

座標：場地 1.4 m、中心在世界原點時，`arena_map.json` 的 (x, y) = 世界的 (x − 0.70, y − 0.70)。原點 (0.185, 0.14) 就是 (−0.515, −0.56)。

控制器參數（`controllerArgs`）就是 `route_follow.py` 的參數，預設 `--hug --return-home`；另外有 `--origin=X,Y`（場地左下角在世界的位置）和 `--cameras`（打開 5 個鏡頭畫面）。

## 不開 Webots 也能測

`Code/Server/route_sim.py` 是同一套邏輯的 2D 模擬，在 Pi 上就能跑，也能讀 Webots 世界裡的障礙物：

```
cd Code/Server
python3 route_sim.py --world ../../webots/worlds/arena_lap.wbt --runs 4
python3 route_sim.py --layout lap --runs 4
python3 route_sim.py --blocks 65,75,8,28 112,132,60,70
```
