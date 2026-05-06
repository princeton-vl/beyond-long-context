from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import shutil
import subprocess

from vidgeom.assets import assign_token_letters


Color = Tuple[int, int, int]
LETTER_SCALE_FACTOR = 0.25


@dataclass
class LetterStyle:
    color: Color
    outline_color: Optional[Color]
    font_scale: float
    thickness: int
    outline_thickness: int
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class LetterRenderPlan:
    width: int
    height: int
    fps: int
    belts: int
    pad_px: float
    lane_centers: List[int]
    background: np.ndarray
    letter_map: Dict[str, str]
    style: LetterStyle
    center_y: int

    @property
    def frame_duration(self) -> float:
        if self.fps <= 0:
            return 1.0
        return 1.0 / float(self.fps)


def _clamp_color(color: Sequence[int]) -> Color:
    return (
        int(max(0, min(255, color[0]))),
        int(max(0, min(255, color[1]))),
        int(max(0, min(255, color[2]))),
    )


def _lane_centers(width: int, belts: int, pad_px: float) -> List[int]:
    belts = max(1, belts)
    usable = max(1.0, float(width) - 2.0 * pad_px)
    lane_w = usable / float(belts)
    centers: List[int] = []
    for lane in range(belts):
        center = pad_px + lane_w * (lane + 0.5)
        centers.append(int(round(center)))
    return centers


def _build_background(width: int, height: int, belts: int, pad_px: float) -> np.ndarray:
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:, :] = (20, 22, 28)
    lane_centers = _lane_centers(width, belts, pad_px)
    if not lane_centers:
        return bg
    usable = max(1.0, float(width) - 2.0 * pad_px)
    lane_w = usable / float(max(1, belts))
    belt_top = int(height * 0.15)
    belt_bottom = int(height * 0.9)
    font = cv2.FONT_HERSHEY_DUPLEX
    # Larger but lighter labels with a subtle outline for clarity
    label_scale = max(0.4, 0.042 * height / 256.0)
    label_scale *= 1.55
    label_thickness = max(1, int(round(label_scale * 4.5)))
    label_outline = max(1, int(round(label_thickness * 0.9)))
    label_y = int(height * 0.072)
    label_shadow_offset = max(1, int(round(height * 0.004)))
    for lane, center in enumerate(lane_centers):
        half_w = lane_w * 0.42
        left = int(round(center - half_w))
        right = int(round(center + half_w))
        cv2.rectangle(bg, (left, belt_top), (right, belt_bottom), (32, 34, 42), thickness=-1)
        # Static stripes to suggest conveyor tread
        stripe_color = (40, 44, 56)
        stripe_height = max(2, int(0.015 * height))
        gap = stripe_height * 2
        y = belt_top + (lane % 2) * stripe_height
        while y < belt_bottom:
            cv2.rectangle(bg, (left, y), (right, min(belt_bottom, y + stripe_height)), stripe_color, thickness=-1)
            y += gap
        # Rails
        rail_color = (80, 84, 92)
        cv2.rectangle(bg, (left - 3, belt_top), (left + 1, belt_bottom), rail_color, thickness=-1)
        cv2.rectangle(bg, (right - 1, belt_top), (right + 3, belt_bottom), rail_color, thickness=-1)
        label_text = str(lane)
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, label_scale, label_thickness)
        label_x = int(center - label_w / 2)
        # Drop shadow / outline for readability against bright belts
        shadow_pos = (label_x + label_shadow_offset, label_y + label_shadow_offset)
        cv2.putText(
            bg,
            label_text,
            shadow_pos,
            font,
            label_scale,
            (10, 12, 18),
            label_outline,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            label_text,
            (label_x, label_y),
            font,
            label_scale,
            (238, 240, 248),
            label_thickness,
            lineType=cv2.LINE_AA,
        )
    return bg


def build_letter_render_plan(
    width: int,
    height: int,
    fps: int,
    belts: int,
    lane_pad: float,
    letter_plan: Dict[str, Any],
) -> LetterRenderPlan:
    pad_px = max(0.0, float(lane_pad)) * float(width)
    style_cfg = letter_plan.get("style", {}) if isinstance(letter_plan, dict) else {}
    color = _clamp_color(style_cfg.get("color", (245, 245, 245)))
    outline = style_cfg.get("outline_color")
    outline_color = None if outline is None else _clamp_color(outline)
    font_scale_cfg = float(style_cfg.get("font_scale", 5.4))
    ref_size = float(style_cfg.get("image_size", 256)) or 256.0
    base_font_scale = font_scale_cfg * (float(height) / ref_size)
    font_scale = base_font_scale * LETTER_SCALE_FACTOR
    base_thickness = int(style_cfg.get("thickness", 14))
    base_outline = int(style_cfg.get("outline_thickness", 18))
    thickness = max(1, int(round(base_thickness * LETTER_SCALE_FACTOR)))
    outline_thickness = max(0, int(round(base_outline * LETTER_SCALE_FACTOR)))
    lane_centers = _lane_centers(width, belts, pad_px)
    background = _build_background(width, height, belts, pad_px)
    center_y = int(height * 0.45)
    style = LetterStyle(
        color=color,
        outline_color=outline_color,
        font_scale=font_scale,
        thickness=max(1, thickness),
        outline_thickness=max(0, outline_thickness),
    )
    return LetterRenderPlan(
        width=int(width),
        height=int(height),
        fps=int(fps),
        belts=max(1, int(belts)),
        pad_px=pad_px,
        lane_centers=lane_centers or [width // 2],
        background=background,
        letter_map=dict(letter_plan.get("map", {})) if isinstance(letter_plan, dict) else {},
        style=style,
        center_y=center_y,
    )


def create_letter_plan(
    tokens: Sequence[str],
    letter_cfg: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    cfg = letter_cfg or {}
    return assign_token_letters(tokens, rng, cfg)


def _lane_for_index(idx: int, lanes: Sequence[str], belts: int) -> int:
    if idx < len(lanes):
        try:
            lane = int(lanes[idx])
        except (ValueError, TypeError):
            lane = 0
    else:
        lane = 0
    return max(0, min(belts - 1, lane))


def _render_letter(frame: np.ndarray, plan: LetterRenderPlan, text: str, lane: int) -> None:
    lane = max(0, min(len(plan.lane_centers) - 1, lane))
    center_x = plan.lane_centers[lane]
    font = plan.style.font_face
    (text_w, text_h), baseline = cv2.getTextSize(text, font, plan.style.font_scale, plan.style.thickness)
    origin_x = int(center_x - text_w / 2)
    origin_y = int(plan.center_y + text_h / 2)
    if plan.style.outline_color is not None and plan.style.outline_thickness > 0:
        cv2.putText(
            frame,
            text,
            (origin_x, origin_y),
            font,
            plan.style.font_scale,
            plan.style.outline_color,
            plan.style.outline_thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        text,
        (origin_x, origin_y),
        font,
        plan.style.font_scale,
        plan.style.color,
        plan.style.thickness,
        lineType=cv2.LINE_AA,
    )


def render_sequence_frames(
    plan: LetterRenderPlan,
    tokens: Sequence[str],
    lanes: Sequence[str],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    frames: List[np.ndarray] = []
    frame_meta: List[Dict[str, any]] = []
    dt = plan.frame_duration
    for idx, raw_token in enumerate(tokens):
        token = str(raw_token)
        lane = _lane_for_index(idx, lanes, plan.belts)
        letter = plan.letter_map.get(token, token)
        frame = plan.background.copy()
        _render_letter(frame, plan, letter, lane)
        frames.append(frame)
        frame_meta.append(
            {
                "frame_index": idx,
                "time": idx * dt,
                "items": [
                    {
                        "token": token,
                        "letter": letter,
                        "lane": lane,
                        "seq_index": idx,
                        "visible": True,
                        "visibility": "onscreen",
                    }
                ],
            }
        )
    return frames, frame_meta


def render_slice_frames(
    plan: LetterRenderPlan,
    tokens: Sequence[str],
    lanes: Sequence[str],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    return render_sequence_frames(plan, tokens, lanes)


def encode_frames_to_mp4(
    frames: Sequence[np.ndarray],
    fps: int,
    out_path: str,
    crf: int = 23,
    preset: str = "veryfast",
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
) -> None:
    if not frames:
        raise RuntimeError("No frames provided for encoding")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    height, width, _ = frames[0].shape
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        pix_fmt,
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                raise RuntimeError("Frame size mismatch during encoding")
            proc.stdin.write(frame.astype(np.uint8).tobytes(order="C"))
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {err[:2000]}")
