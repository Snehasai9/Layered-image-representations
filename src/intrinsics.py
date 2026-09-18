from typing import Tuple
import cv2
import numpy as np

class IntrinsicDecomposer:

    def __init__(self, sigma_spatial: float = 7.0, sigma_range: float = 0.1):
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range

    def decompose(
        self, rgba_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        rgb =rgba_image[:, :, :3].astype(np.float32) / 255.0
        alpha =rgba_image[:, :, 3]

        intensity = cv2.cvtColor(
            (rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        intensity =intensity.astype(np.float32) / 255.0

        shading =cv2.bilateralFilter(
            intensity, d=9, sigmaColor=75, sigmaSpace=75
        )
        shading =np.clip(shading, 0.05, 1.0) 

        shading_3ch = np.repeat(shading[:, :, np.newaxis], 3, axis=2)
        albedo = rgb / shading_3ch
        albedo = np.clip(albedo, 0.0, 1.0)

        albedo_rgba = np.dstack(((albedo * 255).astype(np.uint8), alpha))
        shading_rgba = np.dstack(
            ((shading_3ch * 255).astype(np.uint8), alpha)
        )

        return albedo_rgba, shading_rgba