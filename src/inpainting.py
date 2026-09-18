from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image

class LayerInpainter:
    def __init__(self, inpaint_radius: int = 5):
        self.inpaint_radius =inpaint_radius

    def inpaint_layer(
        self,
        image_rgb: np.ndarray,
        occlusion_mask: np.ndarray,
        layer_mask: np.ndarray,
    ) -> np.ndarray:
        kernel =np.ones((5, 5), dtype=np.uint8)
        dilated_occlusion =cv2.dilate(occlusion_mask.astype(np.uint8), kernel, iterations=2)
        inpainted_bgr =cv2.inpaint(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            dilated_occlusion,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )
        inpainted_rgb =cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        alpha =(layer_mask.astype(np.uint8)) * 255
        rgba =np.dstack((inpainted_rgb, alpha))
        return rgba

    def process_layers(
        self, image: Image.Image, ordered_layers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        image_rgb =np.array(image.convert("RGB"))
        h, w = image_rgb.shape[:2]
        cumulative_occ =np.zeros((h, w), dtype=bool)

        results = []
        for layer in ordered_layers:
            current_mask =layer["mask"]
            is_deepest =layer["depth"]== 1.0 or layer["id"] == -1

            if is_deepest:
                rgba = self.inpaint_layer(
                    image_rgb=image_rgb,
                    occlusion_mask=cumulative_occ,
                    layer_mask=np.ones(
                        (h, w), dtype=bool
                    ),
                )
            else:
                alpha =(current_mask.astype(np.uint8)) *255
                rgba =np.dstack((image_rgb, alpha))
                cumulative_occlusion = np.logical_or(
                    cumulative_occ, current_mask
                )
            layer_da=layer.copy()
            layer_da["rgba"] =rgba
            results.append(layer_da)
        return results