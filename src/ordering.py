from typing import Any, Dict, List
import numpy as np

class LayerOrderer:
    def __init__(self):
        pass
    def compute_layer_depth(self, mask: np.ndarray, depth_map: np.ndarray) -> float:
        masked_depth = depth_map[mask]
        if len(masked_depth) == 0:
            return 1.0
        return float(np.median(masked_depth))

    def order_layers(self,segments: List[Dict[str, Any]],depth_map: np.ndarray,image_shape: tuple,) -> List[Dict[str, Any]]:
        h, w = image_shape[:2]
        ordered_layers = []
        combined_mask = np.zeros((h, w), dtype=bool)

        for seg in segments:
            mask =seg["mask"]
            rep_depth =self.compute_layer_depth(mask, depth_map)
            combined_mask =np.logical_or(combined_mask, mask)
            ordered_layers.append(
                {
                    "id": seg["id"],
                    "label": seg["label"],
                    "mask": mask,
                    "depth": rep_depth,
                }
            )
        bg_mask =~combined_mask
        if np.sum(bg_mask) >100:
            ordered_layers.append(
                {
                    "id": -1,
                    "label": "background_stuff",
                    "mask": bg_mask,
                    "depth": 1.0,
                }
            )
        ordered_layers.sort(key=lambda x: x["depth"])
        for idx, layer in enumerate(ordered_layers):
            layer["order_index"] =idx

        return ordered_layers