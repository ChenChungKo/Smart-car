"""The dashboard web server (Flask): one page, MJPEG streams of the cameras, the map and the bird's-eye view,
and /api/state as JSON. It only reads from the Hub."""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from aruco_localizer import ArucoLocalizer, car_from_camera
from dashboard.hub import Hub
from dashboard.ipm import GroundMapper
from dashboard.live import all_mounts
from dashboard.render import body_polygon, render_map

CAMERAS = ("front", "rear", "left", "right", "gimbal")
DEFAULT_LAYERS = {"trail", "tags", "blocks", "lane", "sensors"}


def wall_gaps(pose: dict, arena_m, body_m) -> dict:
    """Centimetres from the nearest point of the body to each wall, and a sentence about where the car is."""
    w, h = arena_m
    poly = body_polygon(pose["x"], pose["y"], math.radians(pose["yaw_deg"]), body_m[0], body_m[1])
    gaps = {"left": float(poly[:, 0].min()) * 100, "right": (w - float(poly[:, 0].max())) * 100,
            "bottom": float(poly[:, 1].min()) * 100, "top": (h - float(poly[:, 1].max())) * 100}
    names = {"left": "左牆", "right": "右牆", "bottom": "下牆", "top": "上牆"}
    nearest = min(gaps, key=gaps.get)
    heading = pose["yaw_deg"] % 360
    toward = {0: "右牆", 90: "上牆", 180: "左牆", 270: "下牆"}
    nearest_dir = min(toward, key=lambda a: min(abs(heading - a), 360 - abs(heading - a)))
    off = min(abs(heading - nearest_dir), 360 - abs(heading - nearest_dir))
    text = f"最靠近{names[nearest]}（車身離牆 {gaps[nearest]:.1f} cm），車頭朝{toward[nearest_dir]}" + ("" if off < 8 else f"（偏 {off:.0f}°）")
    return {k: round(v, 1) for k, v in gaps.items()}, text


def create_app(hub: Hub, arena: dict, car: dict) -> Flask:
    app = Flask(__name__)
    loc = ArucoLocalizer(arena, car)
    mounts = all_mounts(car)
    mapper = GroundMapper(loc, car_from_camera, tuple(float(v) for v in car.get("body_m", [0.275, 0.17])))

    def mjpeg(make):
        def gen():
            last = None
            while True:
                got = make()
                if got is not None and got[1] != last:
                    last = got[1]
                    data = hub.jpeg(got[2], got[0], got[1])
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                time.sleep(0.08)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/state")
    def state():
        s = hub.snapshot()
        if s.get("pose") and s.get("arena_m") and s.get("body_m"):
            gaps, text = wall_gaps(s["pose"], s["arena_m"], s["body_m"])
            s["wall_gap_cm"], s["where"] = gaps, text
        s["server_time"] = time.time()
        return jsonify(s)

    @app.route("/api/imu/zero", methods=["POST"])
    def imu_zero():
        from dashboard import imu_feed
        if imu_feed.FEED is not None:
            imu_feed.FEED.zero()
        return jsonify(ok=True)

    @app.route("/stream/map")
    def stream_map():
        layers = set((request.args.get("layers") or ",".join(DEFAULT_LAYERS)).split(","))
        box = {"t": 0.0}

        def make():
            time.sleep(0.12)
            s = hub.snapshot()
            img = render_map(s, arena, car, layers)
            box["t"] = time.time()
            return img, box["t"], "map" + ",".join(sorted(layers))
        return mjpeg(make)

    @app.route("/snapshot/bev")
    def snapshot_bev():
        """One bird's-eye picture from the latest frames, made when asked (no stream)."""
        use = [c for c in (request.args.get("cams") or "front,rear,left,right").split(",") if c in CAMERAS]
        items = []
        for name in use:
            got = hub.frame(name)
            if got is not None and time.time() - got[1] < 3.0:
                items.append((name, mounts[name], got[0]))
        if not items:
            return "no camera picture", 503
        ok, buf = cv2.imencode(".jpg", mapper.mosaic(items), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/stream/<name>")
    def stream_camera(name: str):
        if name not in CAMERAS:
            return "unknown camera", 404

        def make():
            got = hub.frame(name)
            return None if got is None else (got[0], got[1], name)
        return mjpeg(make)

    return app


def start_server(hub: Hub, arena: dict, car: dict, host: str = "0.0.0.0", port: int = 5000):
    from werkzeug.serving import make_server
    server = make_server(host, port, create_app(hub, arena, car), threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>車子儀表板</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#1d2330;--mute:#6b7385;--line:#dde1e8;--ok:#1c9a4a;--bad:#d23b3b;--acc:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1d222b;--ink:#e8ebf1;--mute:#98a1b3;--line:#2d3440;--ok:#4fcf7d;--bad:#ff6b6b;--acc:#6fa3ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Noto Sans TC",sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);flex-wrap:wrap}
header h1{font-size:16px;margin:0 8px 0 0}.pill{padding:2px 10px;border-radius:99px;border:1px solid var(--line);font-size:13px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mute{color:var(--mute)}
main{display:grid;grid-template-columns:minmax(300px,520px) minmax(360px,1fr);gap:14px;padding:14px;max-width:1900px;margin:auto}
main.wide{grid-template-columns:1fr}main.wide .camcard{grid-row:1}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--mute);font-weight:600;letter-spacing:.02em}
img.stream{width:100%;border-radius:6px;background:#0b0d11;display:block;min-height:60px}
img#main{max-height:68vh;object-fit:contain}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.tabs button{border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:5px 12px;border-radius:7px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.tabs .sp{flex:1}.tabs button.off{color:var(--mute)}
.opts{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0;font-size:13px}.opts label{display:flex;gap:4px;align-items:center;cursor:pointer}
table{border-collapse:collapse;width:100%}td{padding:2px 6px;border-bottom:1px solid var(--line)}td:first-child{color:var(--mute);width:42%}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media (max-width:700px){.grid2{grid-template-columns:1fr}}
svg text{fill:var(--mute);font-size:10px}pre{margin:0;max-height:150px;overflow:auto;font:12px/1.35 ui-monospace,monospace;color:var(--mute);white-space:pre-wrap}
</style></head><body>
<header><h1>車子儀表板</h1><span class="pill" id="mode">-</span><span class="pill" id="conn">連線中…</span>
<span>電池 <b id="bat">-</b></span><span>CPU <b id="cpu">-</b></span><span>溫度 <b id="temp">-</b></span></header>
<main>
<section>
 <div class="card"><h2>地圖與車子位置（俯視，上＝+y，右＝+x）</h2>
  <img class="stream" id="map" alt="map">
  <div class="opts" id="layers">
   <label><input type="checkbox" value="trail" checked>軌跡</label><label><input type="checkbox" value="tags" checked>標籤</label>
   <label><input type="checkbox" value="blocks" checked>障礙物</label><label><input type="checkbox" value="lane" checked>車道</label>
   <label><input type="checkbox" value="sensors" checked>感測器</label><label><input type="checkbox" value="fov">相機視野</label>
  </div>
  <table><tr><td>位置 x、y</td><td id="pos">-</td></tr><tr><td>車頭朝向</td><td id="yaw">-</td></tr>
  <tr><td>位置來源</td><td id="src">-</td></tr><tr><td>在地圖的哪裡</td><td id="where">-</td></tr>
  <tr><td>車身到牆（cm）</td><td id="gaps">-</td></tr></table>
 </div>
</section>
<section>
 <div class="card camcard"><h2>視角（一次只開一個畫面，其他不傳輸）</h2>
  <div class="tabs" id="tabs"></div>
  <img class="stream" id="main" alt="camera">
  <div class="opts" id="bevopts" style="display:none"><span class="mute">鳥瞰圖（按下當下拍一張，不會更新）使用的相機：</span>
   <label><input type="checkbox" value="front" checked>前</label><label><input type="checkbox" value="rear" checked>後</label>
   <label><input type="checkbox" value="left" checked>左</label><label><input type="checkbox" value="right" checked>右</label>
   <label><input type="checkbox" value="gimbal">雲台</label><button onclick="setMain()">重新拍一張</button></div>
 </div>
 <div class="grid2" style="margin-top:14px">
  <div class="card"><h2>感測器</h2>
   <table><tr><td>超音波</td><td id="sonic">-</td></tr><tr><td>看到的標籤</td><td id="tags">-</td></tr>
   <tr><td>狀態</td><td id="note">-</td></tr></table>
   <svg viewBox="0 0 200 150" width="100%" style="max-height:170px;margin-top:6px" id="car">
    <rect x="62" y="12" width="76" height="126" rx="10" fill="none" stroke="currentColor" opacity=".5"/>
    <path d="M100 4 L108 16 L92 16 Z" fill="currentColor" opacity=".5"/>
    <g id="irdots"></g></svg>
   <div class="mute" style="font-size:12px">紅外線：亮＝有東西（只對反光的表面有反應，灰牆、黑色看不到）</div></div>
  <div class="card"><h2>日誌</h2><pre id="log"></pre></div>
  <div class="card" style="grid-column:1/-1"><h2>IMU（GY-95T）　<span id="imustat" class="mute"></span></h2>
   <div class="grid2"><table>
    <tr><td>加速度 x / y / z（g）</td><td id="imu_acc">-</td></tr>
    <tr><td>陀螺儀 x / y / z（°/s）</td><td id="imu_gyro">-</td></tr>
    <tr><td>模組姿態 roll / pitch / yaw（°）</td><td id="imu_rpy">-</td></tr>
    <tr><td>陀螺儀 z 積分（°）</td><td><b id="imu_int">-</b> <button onclick="fetch('/api/imu/zero',{method:'POST'})">歸零</button>
      <span class="mute">（手轉車子，逆時針為正）</span></td></tr>
    <tr><td>磁力計 x / y / z（高斯）</td><td id="imu_mag">-</td></tr>
    <tr><td>溫度</td><td id="imu_temp">-</td></tr></table>
   <svg viewBox="0 0 200 120" width="100%" style="max-height:140px"><circle cx="60" cy="60" r="48" fill="none" stroke="currentColor" opacity=".35"/>
    <g id="imu_arrow"><path d="M60 16 L68 40 L52 40 Z" fill="#d23b3b"/><line x1="60" y1="60" x2="60" y2="40" stroke="#d23b3b" stroke-width="3"/></g>
    <text x="60" y="12" text-anchor="middle">模組 yaw</text>
    <circle cx="150" cy="60" r="48" fill="none" stroke="currentColor" opacity=".35"/>
    <g id="imu_arrow2"><path d="M150 16 L158 40 L142 40 Z" fill="#2f6fdb"/><line x1="150" y1="60" x2="150" y2="40" stroke="#2f6fdb" stroke-width="3"/></g>
    <text x="150" y="12" text-anchor="middle">陀螺儀積分</text></svg></div></div>
 </div>
</section></main>
<script>
const CAMS=[["map","—"],["front","前"],["rear","後"],["left","左"],["right","右"],["gimbal","雲台"],["bev","鳥瞰"]];
let view="front";
const $=id=>document.getElementById(id);
const IR={front_left:[72,18],front_right:[128,18],wheel_left_front:[56,52],wheel_left_rear:[56,100],wheel_right_front:[144,52],wheel_right_rear:[144,100]};
const IRN={front_left:"左前",front_right:"右前",wheel_left_front:"左前輪",wheel_left_rear:"左後輪",wheel_right_front:"右前輪",wheel_right_rear:"右後輪"};
function layers(){return [...document.querySelectorAll('#layers input:checked')].map(i=>i.value).join(',')||'none'}
function setMap(){$('map').src='/stream/map?layers='+layers()+'&t='+Date.now()}
function bevCams(){return [...document.querySelectorAll('#bevopts input:checked')].map(i=>i.value).join(',')}
function setMain(){
  $('bevopts').style.display=view==='bev'?'flex':'none';
  $('main').style.display=view==='off'?'none':'block';
  if(view==='off')$('main').removeAttribute('src');
  else $('main').src=view==='bev'?'/snapshot/bev?cams='+bevCams()+'&t='+Date.now():'/stream/'+view+'?t='+Date.now();
  document.querySelectorAll('#tabs button[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v===view));
}
CAMS.filter(c=>c[0]!=='map').concat([["off","關閉"]]).forEach(([v,t])=>{const b=document.createElement('button');b.textContent=t;b.dataset.v=v;if(v==='off')b.className='off';b.onclick=()=>{view=v;setMain()};$('tabs').appendChild(b)});
{const sp=document.createElement('span');sp.className='sp';$('tabs').appendChild(sp);const z=document.createElement('button');z.textContent='放大';z.onclick=()=>{const w=document.querySelector('main').classList.toggle('wide');z.textContent=w?'縮小':'放大'};$('tabs').appendChild(z)}
document.querySelectorAll('#layers input').forEach(i=>i.onchange=setMap);
document.querySelectorAll('#bevopts input').forEach(i=>i.onchange=setMain);
Object.entries(IR).forEach(([k,[x,y]])=>{const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.innerHTML='<circle id="ir_'+k+'" cx="'+x+'" cy="'+y+'" r="7" fill="#999"/><text x="'+(x<100?x-34:x+10)+'" y="'+(y+3)+'">'+IRN[k]+'</text>';$('irdots').appendChild(g)});
setMap();setMain();
async function tick(){
  try{const s=await (await fetch('/api/state')).json();
    const age=Date.now()/1000-s.server_time; $('conn').textContent='已連線';$('conn').className='pill ok';
    $('mode').textContent=s.mode==='run'?'跑圈中':'即時監看';
    $('bat').textContent=s.battery_v?s.battery_v.toFixed(2)+' V':'-';$('bat').className=s.battery_v&&s.battery_v<6.9?'bad':'';
    $('cpu').textContent=s.cpu!=null?s.cpu+' %':'-';$('temp').textContent=s.temp_c!=null?s.temp_c+' °C':'-';
    const p=s.pose;
    $('pos').textContent=p?('x '+(p.x*100).toFixed(1)+' cm，y '+(p.y*100).toFixed(1)+' cm'):'（沒有位置）';
    $('yaw').textContent=p?p.yaw_deg.toFixed(1)+'°':'-';
    $('src').textContent=p?(p.source+(p.rms_px!=null?'，殘差 '+p.rms_px+' px':'')):(s.pose_note||'-');
    $('where').textContent=s.where||'-';
    const g=s.wall_gap_cm;$('gaps').textContent=g?('左 '+g.left+'　右 '+g.right+'　下 '+g.bottom+'　上 '+g.top):'-';
    $('sonic').textContent=s.sonic_cm!=null?s.sonic_cm.toFixed(1)+' cm（轉向 '+(s.sonic_pan_deg||0)+'°）':'沒有回應';
    $('tags').textContent=s.tags_seen?Object.entries(s.tags_seen).map(([c,t])=>({front:'前',rear:'後',left:'左',right:'右',gimbal:'雲台'})[c]+':'+(t.length?t.join(','):'無')).join('　'):'-';
    $('note').textContent=s.status||s.pose_note||'-';
    const m=s.imu;
    if(m){const f=(a,d)=>a.map(v=>v.toFixed(d)).join(' / ');
      $('imu_acc').textContent=f(m.acc_g,3);$('imu_gyro').textContent=f(m.gyro_dps,2);$('imu_rpy').textContent=f(m.rpy_deg,2);
      $('imu_int').textContent=m.gyro_yaw_deg.toFixed(1);$('imu_mag').textContent=f(m.mag_gauss,3)+'（總強度 '+Math.hypot(...m.mag_gauss).toFixed(2)+'，校準等級 '+m.mag_level+'）';
      $('imu_temp').textContent=m.temperature_c.toFixed(1)+' °C';
      $('imustat').textContent=m.rate_hz+' 筆/秒，錯誤封包 '+m.bad_frames+(m.age_s>1?'，已停止更新':'');
      $('imu_arrow').setAttribute('transform','rotate('+(-m.rpy_deg[2])+' 60 60)');$('imu_arrow2').setAttribute('transform','rotate('+(-m.gyro_yaw_deg)+' 150 60)')}
    else{$('imustat').textContent='沒有收到資料';['imu_acc','imu_gyro','imu_rpy','imu_int','imu_mag','imu_temp'].forEach(i=>$(i).textContent='-')}
    Object.keys(IR).forEach(k=>{const c=$('ir_'+k);if(c)c.setAttribute('fill',s.ir&&s.ir[k]?'#d23b3b':'#8c93a3')});
    $('log').textContent=(s.log||[]).slice(-14).join('\n');
  }catch(e){$('conn').textContent='連不上';$('conn').className='pill bad'}
}
tick();setInterval(tick,500);
</script></body></html>"""
