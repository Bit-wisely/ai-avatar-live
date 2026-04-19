import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import Delaunay

_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

def _make_detector(static=True):
    base = mp_python.BaseOptions(model_asset_path=os.path.normpath(_MODEL))
    mode = mp_vision.RunningMode.IMAGE if static else mp_vision.RunningMode.VIDEO
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, running_mode=mode, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)

_STATIC_DET = None
_VIDEO_DET = None
_VIDEO_TS = 0

def _get_pts(det, bgr, ts=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = det.detect(img) if ts is None else det.detect_for_video(img, ts)
    if not result.face_landmarks: return None
    return np.array([(p.x * bgr.shape[1], p.y * bgr.shape[0]) for p in result.face_landmarks[0]], dtype=np.float32)

class Portrait3DRenderer:
    def __init__(self, portrait: np.ndarray):
        global _STATIC_DET
        if _STATIC_DET is None: _STATIC_DET = _make_detector(True)
        self.portrait = portrait.copy()
        self.src_pts = _get_pts(_STATIC_DET, portrait)
        self.tri_idx = Delaunay(self.src_pts).simplices if self.src_pts is not None else None
        self.mask = self._make_mask() if self.src_pts is not None else None
        self._smoothed = None

    def _make_mask(self):
        m = np.zeros(self.portrait.shape[:2], dtype=np.float32)
        cv2.fillConvexPoly(m, self.src_pts[_FACE_OVAL].astype(np.int32), 1.0)
        return cv2.GaussianBlur(m, (25, 25), 12)[:, :, np.newaxis]

    def render(self, live: np.ndarray) -> np.ndarray:
        global _VIDEO_DET, _VIDEO_TS
        if self.tri_idx is None: return self.portrait
        if _VIDEO_DET is None: _VIDEO_DET = _make_detector(False)
        _VIDEO_TS += 33
        
        dst_pts = _get_pts(_VIDEO_DET, live, _VIDEO_TS)
        if dst_pts is None: return self.portrait

        dst_scaled = dst_pts * np.array([self.portrait.shape[1]/live.shape[1], self.portrait.shape[0]/live.shape[0]])
        self._smoothed = dst_scaled if self._smoothed is None else 0.35 * dst_scaled + 0.65 * self._smoothed
        
        out = self.portrait.copy()
        for t in self.tri_idx:
            self._warp(out, self.src_pts[t], self._smoothed[t])
        
        return np.clip(out.astype(np.float32) * self.mask + self.portrait.astype(np.float32) * (1 - self.mask), 0, 255).astype(np.uint8)

    def _warp(self, dst, src_tri, dst_tri):
        sr, dr = cv2.boundingRect(src_tri), cv2.boundingRect(dst_tri)
        if dr[2] <= 0 or dr[3] <= 0 or sr[2] <= 0 or sr[3] <= 0: return
        
        M = cv2.getAffineTransform((src_tri - [sr[0], sr[1]]), (dst_tri - [dr[0], dr[1]]))
        patch = cv2.warpAffine(self.portrait[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]], M, (dr[2], dr[3]))
        
        m = np.zeros((dr[3], dr[2]), dtype=np.float32)
        cv2.fillConvexPoly(m, (dst_tri - [dr[0], dr[1]]).astype(np.int32), 1.0)
        
        # Clip to destination boundaries
        y1, y2, x1, x2 = max(0, dr[1]), min(dst.shape[0], dr[1]+dr[3]), max(0, dr[0]), min(dst.shape[1], dr[0]+dr[2])
        ph, pw = y2 - y1, x2 - x1
        if ph <= 0 or pw <= 0: return
        
        roi = dst[y1:y2, x1:x2]
        p_clamped = patch[:ph, :pw]
        m_clamped = m[:ph, :pw, np.newaxis]
        dst[y1:y2, x1:x2] = (roi * (1 - m_clamped) + p_clamped * m_clamped).astype(np.uint8)
