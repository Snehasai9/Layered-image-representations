from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image

class LayerInpainter:
    def __init__(self, inpaint_radius: int = 15):
        self.inpaint_radius = inpaint_radius

    def inpaint_layer(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray, layer_mask: np.ndarray) -> np.ndarray:
       
        hole_mask = (occlusion_mask > 0).astype(np.uint8) * 255

        if np.sum(hole_mask) == 0:
            alpha = (layer_mask.astype(np.uint8)) * 255
            return np.dstack((image_rgb, alpha))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated_mask = cv2.dilate(hole_mask, kernel, iterations=3)

        inpainted_bgr = cv2.inpaint(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            dilated_mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

        alpha = (layer_mask.astype(np.uint8)) * 255
        return np.dstack((inpainted_rgb, alpha))

    def process_layers(self, image: Image.Image, ordered_layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        image_rgb = np.array(image.convert("RGB"))
        h, w = image_rgb.shape[:2]

        cumulative_occlusion = np.zeros((h, w), dtype=bool)
        results = []

        for layer in ordered_layers:
            current_mask = layer["mask"]
            is_deepest = (layer["depth"] == 1.0 or layer["id"] == -1 or layer["label"] == "couch")

            if is_deepest and np.sum(cumulative_occlusion) > 0:
                rgba = self.inpaint_layer(
                    image_rgb=image_rgb,
                    occlusion_mask=cumulative_occlusion,
                    layer_mask=current_mask if layer["id"] != -1 else np.ones((h, w), dtype=bool)
                )
            else:
                alpha = (current_mask.astype(np.uint8)) * 255
                rgba = np.dstack((image_rgb, alpha))
                cumulative_occlusion = np.logical_or(cumulative_occlusion, current_mask)

            layer_data = layer.copy()
            layer_data["rgba"] = rgba
            results.append(layer_data)

        return results
