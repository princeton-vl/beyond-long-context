"""Shared resize / slicing helpers used by every per-model FLOPs estimator.

Each helper replicates the exact preprocessing routine of the upstream model
so that callers can pass arbitrary (H, W) and get the same effective patch
grid / slice count / sub-crop count the real inference pipeline would see.

References (cited verbatim near each helper):
  - smart_resize: transformers.image_processing_qwen2_vl.smart_resize.
  - MiniCPM slice algorithm: openbmb/MiniCPM-V-4_5/image_processing_minicpmv.py
    methods get_sliced_grid + slice_image + get_sliced_images + find_best_resize
    + get_refine_size + ensure_divide.
  - Phi-4-MM HD transform helper exists already in flops_longvila_mimo_phi.py
    (it is left there because it reads several Phi-4-MM-specific constants).

NOTHING in this file is model-specific outside of the docstring citations -- the
upstream functions match across every Qwen-family preprocessor (Qwen2-VL,
Qwen2.5-VL, Qwen3-VL, Qwen3-Omni, GLM-4.5V) and across MiniCPM-V 2.6 / 4.5.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Qwen family / GLM-4.5V smart_resize
# ---------------------------------------------------------------------------
#
# Source (transformers 4.x, identical body for Qwen2-VL, Qwen2.5-VL, Qwen3-VL,
# Qwen3-Omni, GLM-4.5V image processors):
#
#   def smart_resize(height, width, factor=28,
#                    min_pixels=56*56, max_pixels=14*14*4*1280):
#       if max(height, width) / min(height, width) > 200:
#           raise ValueError(...)
#       h_bar = round(height / factor) * factor
#       w_bar = round(width / factor) * factor
#       if h_bar * w_bar > max_pixels:
#           beta = math.sqrt((height * width) / max_pixels)
#           h_bar = max(factor, math.floor(height / beta / factor) * factor)
#           w_bar = max(factor, math.floor(width / beta / factor) * factor)
#       elif h_bar * w_bar < min_pixels:
#           beta = math.sqrt(min_pixels / (height * width))
#           h_bar = math.ceil(height * beta / factor) * factor
#           w_bar = math.ceil(width * beta / factor) * factor
#       return h_bar, w_bar
#
# We replicate the same body here. `factor` is `patch_size * spatial_merge_size`
# for whichever model is calling. The default min/max-pixel envelope matches
# transformers (used by every Qwen-VL family preprocessor); the GLM-4.5V
# preprocessor uses the same envelope (zai-org/GLM-4.5V/preprocessor_config.json
# inherits min_pixels=3136, max_pixels=12845056).

_DEFAULT_MIN_PIXELS = 56 * 56              # 3136
_DEFAULT_MAX_PIXELS = 14 * 14 * 4 * 1280   # 1003520


def smart_resize(height: int, width: int, factor: int,
                 min_pixels: int = _DEFAULT_MIN_PIXELS,
                 max_pixels: int = _DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Snap (height, width) to multiples of ``factor`` per the Qwen-family
    preprocessor's smart_resize. Defaults match transformers' values.

    Returns (h_snapped, w_snapped). For pathological aspect ratios (>200:1)
    we clamp to the 200:1 limit instead of raising, since this helper is used
    for FLOPs accounting only.
    """
    if height <= 0 or width <= 0:
        return factor, factor
    big = max(height, width)
    small = min(height, width)
    if big / small > 200:
        # clamp the smaller dim up to satisfy the ratio (FLOPs-accounting only)
        small = max(1, big // 200)
        if height >= width:
            width = small
        else:
            height = small
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# MiniCPM-V 2.6 / 4.5 multi-slice
# ---------------------------------------------------------------------------
#
# Source: openbmb/MiniCPM-V-4_5/image_processing_minicpmv.py. The relevant
# methods (verbatim line ranges from the live file):
#
#   ensure_divide (155-156):
#       def ensure_divide(self, length, patch_size):
#           return max(round(length / patch_size) * patch_size, patch_size)
#
#   find_best_resize (158-171):
#       def find_best_resize(self, original_size, scale_resolution, patch_size,
#                            allow_upscale=False):
#           width, height = original_size
#           if (width * height > scale_resolution * scale_resolution) or allow_upscale:
#               r = width / height
#               height = int(scale_resolution / math.sqrt(r))
#               width = int(height * r)
#           best_width = self.ensure_divide(width, patch_size)
#           best_height = self.ensure_divide(height, patch_size)
#           return (best_width, best_height)
#
#   get_refine_size (173-193):
#       refine_width  = ensure_divide(width,  grid_x)
#       refine_height = ensure_divide(height, grid_y)
#       grid_width  = refine_width  / grid_x
#       grid_height = refine_height / grid_y
#       best_grid_size = find_best_resize((grid_width, grid_height), ...,
#                                         allow_upscale=True)
#       refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
#
#   get_sliced_grid (225-254):
#       ratio = (W * H) / (scale_resolution^2)
#       multiple = min(ceil(ratio), max_slice_nums)
#       if multiple <= 1: return None     # NO slicing -> 1 image total.
#       candidate_split_grids_nums = [m for m in [multiple-1, multiple,
#                                                  multiple+1]
#                                       if 1 < m <= max_slice_nums]
#       candidate_grids = []
#       for n in candidate_split_grids_nums:
#           for m in 1..n:  if n%m == 0: candidate_grids.append([m, n//m])
#       best_grid = argmin |log_ratio - log(grid_x/grid_y)|
#       return best_grid
#
#   get_sliced_images (259-280):
#       slice_images = [source_image, *patches_grid]
#       i.e. 1 thumbnail + grid_x*grid_y patches when slicing happens, else 1.
#
#   __init__ (line 28-30 etc.):
#       max_slice_nums=9, scale_resolution=448, patch_size=14, slice_mode=True
#
# Each of the (1 + grid_x*grid_y) images is then run through a SigLIP forward
# at its individual (best_height, best_width) -- the thumbnail at
# find_best_resize(original) and each sub-patch at the per-patch best_grid_size
# from get_refine_size. We expose the *exact* per-image (h, w) list so
# callers can sum SigLIP FLOPs over the variable per-slice geometry.

_MCPM_DEFAULT_SCALE_RESOLUTION = 448
_MCPM_DEFAULT_PATCH = 14
_MCPM_DEFAULT_MAX_SLICE_NUMS = 9


def _ensure_divide(length: float, divisor: int) -> int:
    """Verbatim port of MiniCPM ensure_divide (line 155-156)."""
    return int(max(round(length / divisor) * divisor, divisor))


def _mcpm_find_best_resize(width: int, height: int, scale_resolution: int,
                           patch_size: int, allow_upscale: bool) -> tuple[int, int]:
    """Verbatim port of MiniCPM find_best_resize (line 158-171)."""
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / max(1, height)
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = _ensure_divide(width, patch_size)
    best_height = _ensure_divide(height, patch_size)
    return (best_width, best_height)


def _mcpm_get_refine_size(width: int, height: int, grid: tuple[int, int],
                          scale_resolution: int, patch_size: int) -> tuple[int, int]:
    """Verbatim port of MiniCPM get_refine_size (line 173-193). Returns the
    (refine_width, refine_height) and the per-patch (best_w, best_h) used for
    each sub-crop's SigLIP forward."""
    grid_x, grid_y = grid
    refine_width = _ensure_divide(width, grid_x)
    refine_height = _ensure_divide(height, grid_y)
    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y
    best_grid_size = _mcpm_find_best_resize(int(grid_width), int(grid_height),
                                            scale_resolution, patch_size,
                                            allow_upscale=True)
    return best_grid_size  # (best_w, best_h) for each sub-crop


def _mcpm_get_sliced_grid(width: int, height: int, max_slice_nums: int,
                          scale_resolution: int) -> tuple[int, int] | None:
    """Verbatim port of MiniCPM get_sliced_grid (line 225-254). Returns
    [grid_x, grid_y] or None when no slicing is needed."""
    if width <= 0 or height <= 0:
        return None
    log_ratio = math.log(width / height)
    ratio = (width * height) / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None
    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i == 1 or i > max_slice_nums:
            continue
        candidate_split_grids_nums.append(i)
    candidate_grids = []
    for n in candidate_split_grids_nums:
        m = 1
        while m <= n:
            if n % m == 0:
                candidate_grids.append((m, n // m))
            m += 1
    best_grid = (1, 1)
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            min_error = error
            best_grid = grid
    return best_grid


def minicpm_slice_geometry(
    height: int,
    width: int,
    *,
    scale_resolution: int = _MCPM_DEFAULT_SCALE_RESOLUTION,
    patch_size: int = _MCPM_DEFAULT_PATCH,
    max_slice_nums: int = _MCPM_DEFAULT_MAX_SLICE_NUMS,
) -> list[tuple[int, int]]:
    """Return a list of (h, w) for each ViT forward MiniCPM-V will run on a
    single (height, width) frame.

    With slice_mode=True (the canonical MiniCPM-V default), the returned list
    is:
      - 1 entry  : if no slicing is needed (`get_sliced_grid` returned None).
        The single entry is the thumbnail's resized (h, w) -- the source image
        upsampled to the SigLIP base.
      - 1+grid_x*grid_y entries : 1 thumbnail at find_best_resize(original)
        WITHOUT upsampling, plus grid_x*grid_y sub-patches each at the
        get_refine_size best (h, w).

    The (h, w) returned for each entry is the *per-ViT-forward* spatial size,
    so the caller multiplies (h//patch) * (w//patch) to get the sequence
    length for that forward.

    Source: openbmb/MiniCPM-V-4_5/image_processing_minicpmv.py:
      get_sliced_images (line 259-280): returns [source_image, *patches].
      slice_image       (line 209-233): produces source_image + grid patches.
      The thumbnail with no slicing uses ``find_best_resize(allow_upscale=True)``
      (line 217-222); with slicing uses ``find_best_resize(allow_upscale=False)``
      (line 225) for the source image and ``get_refine_size`` for the patches.
    """
    grid = _mcpm_get_sliced_grid(width, height, max_slice_nums, scale_resolution)
    if grid is None:
        # Single image; the source is upsampled to scale_resolution.
        bw, bh = _mcpm_find_best_resize(width, height, scale_resolution,
                                        patch_size, allow_upscale=True)
        return [(bh, bw)]
    # Slicing path: thumbnail at find_best_resize(original) + grid patches.
    th_w, th_h = _mcpm_find_best_resize(width, height, scale_resolution,
                                        patch_size, allow_upscale=False)
    grid_x, grid_y = grid
    sub_w, sub_h = _mcpm_get_refine_size(width, height, (grid_x, grid_y),
                                         scale_resolution, patch_size)
    out: list[tuple[int, int]] = [(th_h, th_w)]
    for _ in range(grid_x * grid_y):
        out.append((sub_h, sub_w))
    return out


def minicpm_n_slices(height: int, width: int,
                     max_slice_nums: int = _MCPM_DEFAULT_MAX_SLICE_NUMS,
                     scale_resolution: int = _MCPM_DEFAULT_SCALE_RESOLUTION
                     ) -> int:
    """Total ViT forwards per frame (1 thumbnail + grid_x*grid_y sub-crops)."""
    g = _mcpm_get_sliced_grid(width, height, max_slice_nums, scale_resolution)
    if g is None:
        return 1
    return 1 + g[0] * g[1]


__all__ = [
    "smart_resize",
    "minicpm_slice_geometry",
    "minicpm_n_slices",
]
