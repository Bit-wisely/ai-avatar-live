"""
Legacy AvatarRenderer stub – kept for backwards compatibility.
The active renderer is now Portrait3DRenderer in portrait_3d_renderer.py
"""
import cv2
import numpy as np

class AvatarRenderer:
    """Thin wrapper retained for config compatibility."""
    def __init__(self, avatar_path="", model_path="", use_gpu=True):
        self.avatar_image = None
        if avatar_path:
            img = cv2.imread(avatar_path)
            if img is not None:
                self.avatar_image = img
        print("[AvatarRenderer] Legacy stub loaded (use Portrait3DRenderer for 3D output)")

    def render(self, landmarks=None):
        return self.avatar_image if self.avatar_image is not None else np.zeros((512, 512, 3), dtype=np.uint8)
