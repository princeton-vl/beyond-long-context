from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import cv2

from .draw import DrawCmd, DrawList, Prototype, CirclePrim, PolyPrim, LinePrim, ImagePrim, Color
from .assets import AssetStore

def _rgb_to_bgr(c: Color) -> Tuple[int, int, int]:
    return (int(c[2]), int(c[1]), int(c[0]))

def _to_px(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    return (int(round(x * (w - 1))), int(round(y * (h - 1))))

def _apply_transform(points: np.ndarray, tx: float, ty: float, s: float, rot: float) -> np.ndarray:
    # points: (N,2) local
    c = math.cos(rot)
    si = math.sin(rot)
    R = np.array([[c, -si], [si, c]], dtype=np.float32)
    out = (points.astype(np.float32) * s) @ R.T
    out[:, 0] += tx
    out[:, 1] += ty
    return out

@dataclass
class RenderConfig:
    width: int
    height: int

class GeometryRenderer:
    """Rasterizes DrawList into RGB uint8 frames using OpenCV + numpy."""
    def __init__(self, asset_store: AssetStore, cfg: RenderConfig):
        self.assets = asset_store
        self.cfg = cfg

    def render(self, draw: DrawList) -> np.ndarray:
        w, h = self.cfg.width, self.cfg.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # stable sort by z, preserving original order for equal z
        draw_sorted = sorted(enumerate(draw), key=lambda x: (x[1].z, x[0]))
        for _, cmd in draw_sorted:
            self._draw_cmd(frame, cmd)
        return frame  # RGB

    def _draw_cmd(self, frame: np.ndarray, cmd: DrawCmd) -> None:
        kind = cmd.kind
        d = cmd.data
        if kind == "fill":
            color = d.get("color", (0, 0, 0))
            frame[:, :, :] = np.array(color, dtype=np.uint8)[None, None, :]
            return

        if kind == "gradient":
            # linear gradient in rect (x,y,w,h) in normalized coords
            x, y, rw, rh = d["rect"]
            c0 = np.array(d["c0"], dtype=np.float32)
            c1 = np.array(d["c1"], dtype=np.float32)
            vertical = bool(d.get("vertical", True))
            x0, y0 = _to_px(x, y, frame.shape[1], frame.shape[0])
            x1, y1 = _to_px(x + rw, y + rh, frame.shape[1], frame.shape[0])
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 <= x0 or y1 <= y0:
                return
            region = frame[y0:y1, x0:x1, :].astype(np.float32)
            if vertical:
                t = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None]
                grad = c0[None, None, :] * (1 - t[:, :, None]) + c1[None, None, :] * (t[:, :, None])
                grad = np.repeat(grad, x1 - x0, axis=1)
            else:
                t = np.linspace(0, 1, x1 - x0, dtype=np.float32)[None, :]
                grad = c0[None, None, :] * (1 - t[:, :, None]) + c1[None, None, :] * (t[:, :, None])
                grad = np.repeat(grad, y1 - y0, axis=0)
            frame[y0:y1, x0:x1, :] = np.clip(grad, 0, 255).astype(np.uint8)
            return

        if kind == "stripe_fill":
            # Fill a rect with moving diagonal stripes (purely procedural)
            x, y, rw, rh = d["rect"]
            c0 = np.array(d["c0"], dtype=np.uint8)
            c1 = np.array(d["c1"], dtype=np.uint8)
            period = float(d.get("period", 0.08))   # in normalized units
            width = float(d.get("width", 0.035))
            angle = float(d.get("angle", math.pi/4))
            phase = float(d.get("phase", 0.0))
            x0, y0 = _to_px(x, y, frame.shape[1], frame.shape[0])
            x1, y1 = _to_px(x + rw, y + rh, frame.shape[1], frame.shape[0])
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 <= x0 or y1 <= y0:
                return
            H = y1 - y0
            W = x1 - x0
            # Build coordinates in normalized space inside rect
            xs = (np.arange(W, dtype=np.float32) / max(1, (frame.shape[1]-1)))
            ys = (np.arange(H, dtype=np.float32) / max(1, (frame.shape[0]-1)))
            X, Y = np.meshgrid(xs, ys)
            # rotate coords for stripes
            ca, sa = math.cos(angle), math.sin(angle)
            U = X * ca + Y * sa + phase
            # stripes pattern
            m = np.mod(U, period)
            mask = (m < width).astype(np.uint8)[:, :, None]
            region = frame[y0:y1, x0:x1, :]
            out = region * (1 - mask) + (c1[None, None, :] * mask)
            # base fill c0 first (for consistent look)
            base = c0[None, None, :]
            out = base * (1 - mask) + (c1[None, None, :] * mask)
            frame[y0:y1, x0:x1, :] = out.astype(np.uint8)
            return

        if kind == "rect":
            x, y, rw, rh = d["rect"]
            color = d["color"]
            x0, y0 = _to_px(x, y, frame.shape[1], frame.shape[0])
            x1, y1 = _to_px(x + rw, y + rh, frame.shape[1], frame.shape[0])
            cv2.rectangle(frame, (x0, y0), (x1, y1), _rgb_to_bgr(color), thickness=-1)
            return

        if kind == "circle":
            cx, cy = d["center"]
            r = d["radius"]
            color = d["color"]
            px, py = _to_px(cx, cy, frame.shape[1], frame.shape[0])
            pr = int(round(r * min(frame.shape[0], frame.shape[1])))
            cv2.circle(frame, (px, py), pr, _rgb_to_bgr(color), thickness=-1, lineType=cv2.LINE_AA)
            return

        if kind == "poly":
            pts = d["points"]
            color = d["color"]
            arr = np.array([_to_px(x, y, frame.shape[1], frame.shape[0]) for x, y in pts], dtype=np.int32)
            cv2.fillPoly(frame, [arr], _rgb_to_bgr(color), lineType=cv2.LINE_AA)
            return

        if kind == "line":
            p0 = d["p0"]
            p1 = d["p1"]
            width = float(d.get("width", 1.0))
            color = d["color"]
            x0, y0 = _to_px(p0[0], p0[1], frame.shape[1], frame.shape[0])
            x1, y1 = _to_px(p1[0], p1[1], frame.shape[1], frame.shape[0])
            thick = max(1, int(round(width * min(frame.shape[0], frame.shape[1]))))
            cv2.line(frame, (x0, y0), (x1, y1), _rgb_to_bgr(color), thickness=thick, lineType=cv2.LINE_AA)
            return

        if kind == "text":
            pos = d["pos"]
            text = d["text"]
            color = d.get("color", (255, 255, 255))
            scale = float(d.get("scale", 0.5))
            thickness = int(d.get("thickness", 1))
            x, y = _to_px(pos[0], pos[1], frame.shape[1], frame.shape[0])
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _rgb_to_bgr(color), thickness, cv2.LINE_AA)
            return

        if kind == "instance":
            asset_id = d["asset_id"]
            tx, ty = d["pos"]
            s = float(d.get("scale", 1.0))
            rot = float(d.get("rot", 0.0))
            proto = self.assets.get(asset_id)
            self._draw_prototype(frame, proto, tx, ty, s, rot)
            return

        raise ValueError(f"Unknown DrawCmd kind: {kind}")

    def _draw_prototype(self, frame: np.ndarray, proto: Prototype, tx: float, ty: float, s: float, rot: float):
        w, h = frame.shape[1], frame.shape[0]
        for prim in sorted(proto.primitives, key=lambda p: getattr(p, "z", 0.0)):
            if isinstance(prim, CirclePrim):
                pt = np.array([[prim.center[0], prim.center[1]]], dtype=np.float32)
                pt2 = _apply_transform(pt, tx, ty, s, rot)[0]
                px, py = _to_px(float(pt2[0]), float(pt2[1]), w, h)
                pr = int(round(float(prim.radius) * s * min(w, h)))
                cv2.circle(frame, (px, py), max(1, pr), _rgb_to_bgr(prim.color), thickness=-1, lineType=cv2.LINE_AA)
            elif isinstance(prim, PolyPrim):
                pts = np.array(prim.points, dtype=np.float32)
                pts2 = _apply_transform(pts, tx, ty, s, rot)
                arr = np.array([_to_px(float(x), float(y), w, h) for x, y in pts2], dtype=np.int32)
                cv2.fillPoly(frame, [arr], _rgb_to_bgr(prim.color), lineType=cv2.LINE_AA)
            elif isinstance(prim, LinePrim):
                pts = np.array([[prim.p0[0], prim.p0[1]], [prim.p1[0], prim.p1[1]]], dtype=np.float32)
                pts2 = _apply_transform(pts, tx, ty, s, rot)
                x0, y0 = _to_px(float(pts2[0,0]), float(pts2[0,1]), w, h)
                x1, y1 = _to_px(float(pts2[1,0]), float(pts2[1,1]), w, h)
                thick = max(1, int(round(float(prim.width) * s * min(w, h))))
                cv2.line(frame, (x0, y0), (x1, y1), _rgb_to_bgr(prim.color), thickness=thick, lineType=cv2.LINE_AA)
            elif isinstance(prim, ImagePrim):
                self._draw_sprite(frame, prim, tx, ty, s)
            else:
                raise ValueError(f"Unknown primitive type: {type(prim)}")

    def _draw_sprite(self, frame: np.ndarray, prim: ImagePrim, tx: float, ty: float, scale: float) -> None:
        if prim.pixels is None or prim.pixels.ndim != 3 or prim.pixels.shape[2] < 3:
            return
        h_frame, w_frame = frame.shape[0], frame.shape[1]
        center_x, center_y = _to_px(tx, ty, w_frame, h_frame)
        target_w = max(1, int(round(prim.width * scale * w_frame)))
        target_h = max(1, int(round(prim.height * scale * h_frame)))
        if target_w <= 0 or target_h <= 0:
            return

        sprite = prim.pixels
        if sprite.shape[2] == 3:
            alpha_channel = np.full((sprite.shape[0], sprite.shape[1], 1), 255, dtype=np.uint8)
            sprite = np.concatenate([sprite, alpha_channel], axis=2)

        sprite_resized = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        x0 = center_x - target_w // 2
        y0 = center_y - target_h // 2
        x1 = x0 + target_w
        y1 = y0 + target_h

        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(w_frame, x1)
        clip_y1 = min(h_frame, y1)
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            return

        crop_x0 = clip_x0 - x0
        crop_y0 = clip_y0 - y0
        crop_x1 = crop_x0 + (clip_x1 - clip_x0)
        crop_y1 = crop_y0 + (clip_y1 - clip_y0)

        sprite_region = sprite_resized[crop_y0:crop_y1, crop_x0:crop_x1, :]
        rgb = sprite_region[:, :, :3].astype(np.float32)
        alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
        dest = frame[clip_y0:clip_y1, clip_x0:clip_x1, :].astype(np.float32)
        blended = rgb * alpha + dest * (1.0 - alpha)
        frame[clip_y0:clip_y1, clip_x0:clip_x1, :] = np.clip(blended, 0, 255).astype(np.uint8)
