"""
Portrait3DRenderer
==================
Deforms an uploaded portrait to match the live user's head pose and facial
expression using MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10).

Pipeline per frame
------------------
1. Detect 478 face landmarks on the SOURCE portrait  (done once at load)
2. Detect 478 face landmarks on the LIVE webcam frame
3. Build a Delaunay triangulation on the portrait landmarks
4. For every triangle: compute affine transform  portrait → live positions
5. Warp each triangle into an output canvas (same size as the portrait)
6. Composite result over a clean background with edge-alpha mask
7. Apply subtle depth-shading based on the transformation matrix normal
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import Delaunay

# ── Model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "face_landmarker.task"
)
_MODEL_PATH = os.path.normpath(_MODEL_PATH)


def _make_landmarker(static: bool) -> mp_vision.FaceLandmarker:
    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    running_mode = mp_vision.RunningMode.IMAGE if static else mp_vision.RunningMode.VIDEO
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# Module-level landmarkers (one per mode)
_STATIC_LANDMARKER: mp_vision.FaceLandmarker | None = None
_VIDEO_LANDMARKER:  mp_vision.FaceLandmarker | None = None


def _get_static_landmarker() -> mp_vision.FaceLandmarker:
    global _STATIC_LANDMARKER
    if _STATIC_LANDMARKER is None:
        _STATIC_LANDMARKER = _make_landmarker(static=True)
    return _STATIC_LANDMARKER


def _get_video_landmarker() -> mp_vision.FaceLandmarker:
    global _VIDEO_LANDMARKER
    if _VIDEO_LANDMARKER is None:
        _VIDEO_LANDMARKER = _make_landmarker(static=False)
    return _VIDEO_LANDMARKER


# Face-oval landmark indices for alpha mask
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172,  58, 132,  93, 234, 127, 162,  21,  54,
    103,  67, 109,  10
]

# Timestamp counter for VIDEO mode
_video_ts_ms: int = 0


def _landmarks_to_array(result, img_w: int, img_h: int) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in lm],
        dtype=np.float32
    )


def _detect_static(bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_static_landmarker().detect(mp_img)
    return _landmarks_to_array(result, bgr.shape[1], bgr.shape[0])


def _detect_live(bgr: np.ndarray) -> np.ndarray | None:
    """Detect using VIDEO running mode (supports tracking across frames)."""
    global _video_ts_ms
    _video_ts_ms += 33   # approximate 30 fps
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_video_landmarker().detect_for_video(mp_img, _video_ts_ms)
    return _landmarks_to_array(result, bgr.shape[1], bgr.shape[0])


# ── Delaunay triangulation ────────────────────────────────────────────────────

def _build_triangulation(pts: np.ndarray) -> np.ndarray:
    tri = Delaunay(pts)
    return tri.simplices.astype(np.int32)


# ── Alpha mask ────────────────────────────────────────────────────────────────

def _face_oval_mask(shape: tuple, pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.float32)
    oval = pts[_FACE_OVAL].astype(np.int32)
    cv2.fillConvexPoly(mask, oval, 1.0)
    mask = cv2.GaussianBlur(mask, (25, 25), 12)
    return np.clip(mask, 0, 1)


# ── Triangle warping ──────────────────────────────────────────────────────────

def _warp_triangle(
    src: np.ndarray, dst: np.ndarray,
    src_tri: np.ndarray, dst_tri: np.ndarray
):
    sr = cv2.boundingRect(src_tri.astype(np.float32))
    dr = cv2.boundingRect(dst_tri.astype(np.float32))
    H, W = dst.shape[:2]

    # Clamp destination rect
    dx = max(dr[0], 0);  dy = max(dr[1], 0)
    dw = min(dr[0]+dr[2], W) - dx
    dh = min(dr[1]+dr[3], H) - dy
    if dw <= 0 or dh <= 0:
        return

    # Source rect (clamped)
    sx = max(sr[0], 0);  sy = max(sr[1], 0)
    sw = min(sr[0]+sr[2], src.shape[1]) - sx
    sh = min(sr[1]+sr[3], src.shape[0]) - sy
    if sw <= 0 or sh <= 0:
        return

    src_offset = src_tri - np.array([sr[0], sr[1]], dtype=np.float32)
    dst_offset = dst_tri - np.array([dr[0], dr[1]], dtype=np.float32)

    M = cv2.getAffineTransform(
        src_offset[:3].astype(np.float32),
        dst_offset[:3].astype(np.float32)
    )

    src_patch = src[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]]
    if src_patch.size == 0:
        return
    warped = cv2.warpAffine(
        src_patch, M, (dr[2], dr[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Triangle mask in destination patch
    tri_mask = np.zeros((dr[3], dr[2]), dtype=np.float32)
    cv2.fillConvexPoly(tri_mask, dst_offset[:3].astype(np.int32), 1.0)
    m3 = tri_mask[:, :, np.newaxis]

    reg  = dst[dy:dy+dh, dx:dx+dw]
    war  = warped[:dh, :dw]
    msk  = m3[:dh, :dw]
    if reg.shape == war.shape:
        dst[dy:dy+dh, dx:dx+dw] = np.clip(
            reg * (1 - msk) + war * msk, 0, 255
        ).astype(np.uint8)


# ── Depth shading ─────────────────────────────────────────────────────────────

def _shade(image: np.ndarray, transform_mat: np.ndarray | None) -> np.ndarray:
    if transform_mat is None or transform_mat.shape[0] < 3:
        return image
    try:
        R = np.array(transform_mat)[:3, :3]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        factor = float(np.clip(0.85 + 0.18 * np.cos(yaw), 0.55, 1.08))
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    except Exception:
        return image


# ── Main renderer class ───────────────────────────────────────────────────────

class Portrait3DRenderer:
    """
    Initialise once with the uploaded portrait, then call `render(live_frame)`
    every frame to get the deformed portrait output.
    """

    def __init__(self, portrait_bgr: np.ndarray):
        self.portrait = portrait_bgr.copy()
        h, w = portrait_bgr.shape[:2]
        self.size = (w, h)

        print(f"[Portrait3DRenderer] Detecting landmarks in portrait ({w}×{h})…")
        self.src_pts = _detect_static(portrait_bgr)
        self.src_tri_idx: np.ndarray | None = None
        self.face_mask: np.ndarray = np.ones((h, w), dtype=np.float32)

        if self.src_pts is not None:
            self.src_tri_idx = _build_triangulation(self.src_pts)
            self.face_mask = _face_oval_mask(portrait_bgr.shape, self.src_pts)
            print(f"[Portrait3DRenderer] [OK] {len(self.src_pts)} landmarks, {len(self.src_tri_idx)} triangles")
        else:
            print("[Portrait3DRenderer] [WARN] No face detected in portrait - will display static image")

        # Temporal smoothing
        self._smoothed: np.ndarray | None = None
        self._alpha = 0.35      # exponential moving average weight

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(self, live_bgr: np.ndarray) -> np.ndarray:
        """
        Deform the portrait to mimic the pose & expression in `live_bgr`.
        Returns a BGR image of size self.size.
        """
        if self.src_pts is None or self.src_tri_idx is None:
            return self.portrait.copy()

        dst_pts = _detect_live(live_bgr)
        if dst_pts is None:
            # No face – return neutral portrait
            self._smoothed = None
            return self.portrait.copy()

        # Scale dst_pts from live-frame pixel space → portrait pixel space
        lh, lw = live_bgr.shape[:2]
        pw, ph = self.size
        dst_scaled = dst_pts * np.array([pw / lw, ph / lh], dtype=np.float32)

        # Temporal smoothing
        if self._smoothed is None:
            self._smoothed = dst_scaled.copy()
        else:
            self._smoothed = self._alpha * dst_scaled + (1 - self._alpha) * self._smoothed
        dst_use = self._smoothed

        # Warp each triangle
        output = self.portrait.copy()
        for tri_idx in self.src_tri_idx:
            _warp_triangle(self.portrait, output, self.src_pts[tri_idx], dst_use[tri_idx])

        # Blend warped face with original background using face mask
        m3 = self.face_mask[:, :, np.newaxis]
        output = np.clip(
            output.astype(np.float32) * m3 + self.portrait.astype(np.float32) * (1 - m3),
            0, 255
        ).astype(np.uint8)

        return output
