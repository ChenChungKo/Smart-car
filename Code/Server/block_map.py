"""Obstacle blocks remembered across scans.

Every scan reports the blocks it saw (see perceive_map.find_blocks). This map keeps one record per
block and folds each new sighting in, so a block judged wrongly from far away is put right once
the car sees it from closer or from another side. It is saved to a JSON file, so it carries on
across runs as long as nobody moves the foam.

How a sighting updates a block:
- Matching: the sighting lies within a few cm of the block and has its colour; one lying on top of it
  (overlapping) is the same block whatever its colour, since the colour read from a far or
  sidelong view can be wrong. A block's colour is the one it was seen in most often.
- Position: a Kalman update. A sighting's error grows with the square of its range (the gimbal
  camera sits 10 cm off the floor), and is larger when the car was placed by a single tag.
- Heading: only a better sighting may change it. Seeing two faces (an L corner) beats seeing one;
  between equals the nearer one wins. A far one-face guess never overwrites a settled heading.
- A block known only by its raw points becomes a box once a sighting is good enough to box it.
- A block seen once and only from far off (over CONFIRM_REACH_M) is provisional: a shadow or a far
  misread looks like that, so the planner ignores it until a second sighting or a near one. So is
  one only ever seen as scattered dots (no straight foot line), unless the ultrasonic hit it.
- A block that should have been seen (its floor was seen empty this scan) loses a life; after two
  such scans in a row it is removed, being a misread or foam that was taken away.
Blocks outside a scan's view are left as they are.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parent
MAP_FILE = SERVER / "perceive_debug" / "block_map.json"
MATCH_M = 0.06          # a sighting this close to a block's outline (or its points) is that block
MATCH_CENTRE_M = 0.12   # or this close centre to centre
OVERLAP_M = 0.03        # a sighting this close is the same block even in another colour
SAME_HEADING_DEG = 30.0
MISSES_TO_DROP = 2
CONFIRM_REACH_M = 0.45  # one sighting nearer than this counts at once; farther ones need a second


def rect_corners(cx: float, cy: float, yaw: float, size) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    half_l, half_w = size[0] / 2.0, size[1] / 2.0
    return np.array([[cx + c * lx - s * ly, cy + s * lx + c * ly]
                     for lx, ly in ((half_l, half_w), (-half_l, half_w), (-half_l, -half_w), (half_l, -half_w))])


def rect_distance(points: np.ndarray, cx: float, cy: float, yaw: float, size) -> np.ndarray:
    """Signed distance of each point to the rectangle outline (negative inside)."""
    c, s = math.cos(yaw), math.sin(yaw)
    d = points - (cx, cy)
    local = np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])
    q = np.abs(local) - (size[0] / 2.0, size[1] / 2.0)
    return np.hypot(np.maximum(q[:, 0], 0), np.maximum(q[:, 1], 0)) + np.minimum(q.max(axis=1), 0)


def sighting_sigma(range_m: float, tags: int) -> float:
    """Position error (m) of one sighting: about 2.5 cm at 30 cm, 6 cm at 60 cm, more on one tag."""
    sigma = 0.015 + 0.12 * range_m ** 2
    return math.hypot(sigma, 0.02) if tags < 2 else sigma


def heading_gap(a: float, b: float) -> float:
    """Angle between two block headings (deg); a rectangle is the same turned by 180."""
    d = abs(math.degrees(a - b)) % 180.0
    return min(d, 180.0 - d)


class BlockMap:
    def __init__(self, size, path: Path = MAP_FILE):
        self.size = [float(v) for v in size]
        self.path = Path(path)
        self.blocks: list[dict] = []
        self.next_id = 1

    @classmethod
    def load(cls, size, path: Path = MAP_FILE) -> "BlockMap":
        bmap = cls(size, path)
        if bmap.path.exists():
            data = json.loads(bmap.path.read_text(encoding="utf-8"))
            bmap.blocks = data.get("blocks", [])
            bmap.next_id = int(data.get("next_id", len(bmap.blocks) + 1))
        return bmap

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"size_m": self.size, "next_id": self.next_id, "blocks": self.blocks}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---- geometry of a remembered block ----------------------------------

    def footprint(self, block: dict) -> np.ndarray:
        """Points that stand for the block: its rectangle corners and edges, or its raw points."""
        if not block["shaped"]:
            return np.asarray(block["points"], dtype=float).reshape(-1, 2)
        corners = rect_corners(*block["center"], block["yaw"], self.size)
        edges = [a + (b - a) * t for a, b in zip(corners, np.roll(corners, -1, axis=0)) for t in np.linspace(0, 1, 6)]
        return np.asarray(edges)

    def gap_to(self, block: dict, points: np.ndarray) -> float:
        """How far a sighting's points are from a remembered block (0 when touching or inside)."""
        if block["shaped"]:
            return float(max(rect_distance(points, *block["center"], block["yaw"], self.size).min(), 0.0))
        own = self.footprint(block)
        return float(np.hypot(points[:, None, 0] - own[None, :, 0], points[:, None, 1] - own[None, :, 1]).min())

    # ---- update ------------------------------------------------------------

    def update(self, sightings: list[dict], camera_xy, tags: int, seen_free) -> dict:
        """Fold one scan's sightings in. `seen_free(points)` is the share of those floor points this
        scan saw empty. Returns how many blocks were added, updated and removed."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cam = np.asarray(camera_xy, dtype=float)
        matched: set[int] = set()
        counts = {"new": 0, "updated": 0, "removed": 0}
        for sight in sightings:
            pts = np.asarray(sight["points"], dtype=float).reshape(-1, 2)
            reach = float(np.hypot(*(pts - cam).T).min())
            centre = np.asarray(sight["center"] if sight["shaped"] else pts.mean(axis=0))
            best, best_gap = None, math.inf
            for k, block in enumerate(self.blocks):
                if k in matched:
                    continue
                gap = self.gap_to(block, pts)
                if block["color"] == sight["color"]:
                    near = gap <= MATCH_M or math.hypot(*(centre - block["center"])) <= MATCH_CENTRE_M
                else:
                    near = gap <= OVERLAP_M
                if near and gap < best_gap:
                    best, best_gap = k, gap
            if best is None:
                self.blocks.append(self._new(sight, pts, centre, reach, tags, stamp))
                matched.add(len(self.blocks) - 1)
                counts["new"] += 1
            else:
                self._fold(self.blocks[best], sight, pts, centre, reach, tags, stamp)
                matched.add(best)
                counts["updated"] += 1
        kept = []
        for k, block in enumerate(self.blocks):
            if k not in matched and seen_free(self.footprint(block)) >= 0.3:
                block["misses"] += 1
                if block["misses"] >= MISSES_TO_DROP:
                    counts["removed"] += 1
                    continue
            kept.append(block)
        self.blocks = kept
        return counts

    def _new(self, sight: dict, pts, centre, reach: float, tags: int, stamp: str) -> dict:
        block = {"id": self.next_id, "color": sight["color"], "votes": {sight["color"]: 1},
                 "shaped": bool(sight["shaped"]),
                 "center": [float(v) for v in centre], "var": sighting_sigma(reach, tags) ** 2,
                 "yaw": float(sight["yaw"]) if sight["shaped"] else None,
                 "heading_quality": self._quality(sight), "heading_reach": reach if sight["shaped"] else None,
                 "points": np.round(pts, 3).tolist(), "points_reach": reach, "why": sight.get("why"),
                 "hits": 1, "misses": 0, "sonic": int(bool(sight.get("confirmed"))),
                 "solid": int(not sight.get("weak")), "seen": stamp}
        self.next_id += 1
        return block

    @staticmethod
    def _quality(sight: dict) -> int:
        """0: no heading, 1: one face seen near enough, 2: an L corner (two faces) seen."""
        if not sight["shaped"]:
            return 0
        return 1 if sight.get("one_face") else 2

    def _fold(self, block: dict, sight: dict, pts, centre, reach: float, tags: int, stamp: str) -> None:
        block["hits"] += 1
        block["misses"] = 0
        votes = block.setdefault("votes", {block["color"]: block["hits"] - 1})
        votes[sight["color"]] = votes.get(sight["color"], 0) + 1
        block["color"] = max(votes, key=votes.get)
        block["seen"] = stamp
        block["sonic"] += int(bool(sight.get("confirmed")))
        block["solid"] = block.get("solid", 1) + int(not sight.get("weak"))
        if reach < block["points_reach"]:
            # The nearer view of the raw points is the better one.
            block["points"], block["points_reach"], block["why"] = np.round(pts, 3).tolist(), reach, sight.get("why")
        if not sight["shaped"]:
            return
        quality = self._quality(sight)
        noise = sighting_sigma(reach, tags) ** 2
        better = (not block["shaped"] or quality > block["heading_quality"]
                  or (quality == block["heading_quality"] and reach < block["heading_reach"] - 0.05))
        agrees = block["shaped"] and heading_gap(sight["yaw"], block["yaw"]) <= SAME_HEADING_DEG
        if agrees:
            # Same heading: average the position, weighted by how sure each one is.
            gain = block["var"] / (block["var"] + noise)
            block["center"] = [float(c + gain * (z - c)) for c, z in zip(block["center"], centre)]
            block["var"] = (1.0 - gain) * block["var"]
            if better:
                block["yaw"], block["heading_quality"], block["heading_reach"] = float(sight["yaw"]), quality, reach
        elif better:
            # A better look disagrees with the old heading: the old box was laid wrong, start over from it.
            block.update(shaped=True, center=[float(v) for v in centre], var=noise, yaw=float(sight["yaw"]),
                         heading_quality=quality, heading_reach=reach)
        # A worse look that disagrees is ignored: a far one-face guess does not undo a settled heading.

    # ---- for drawing and reporting -----------------------------------------

    def as_blocks(self) -> list[dict]:
        """The remembered blocks in the shape perceive_map draws and reports."""
        out = []
        for block in self.blocks:
            item = {"id": block["id"], "color": block["color"], "shaped": block["shaped"],
                    "points": np.asarray(block["points"], dtype=float).reshape(-1, 2),
                    "confirmed": block["sonic"] > 0, "why": block["why"], "hits": block["hits"],
                    "provisional": (block["hits"] < 2 and block["points_reach"] > CONFIRM_REACH_M)
                                   or (block.get("solid", 1) == 0 and block["sonic"] == 0),
                    "sigma": math.sqrt(block["var"]), "heading_quality": block["heading_quality"]}
            if block["shaped"]:
                item.update(center=tuple(block["center"]), yaw=block["yaw"],
                            corners=rect_corners(*block["center"], block["yaw"], self.size))
            out.append(item)
        return out
