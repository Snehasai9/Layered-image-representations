import numpy as np
import torch
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    def __init__(self, device: str = None):
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1
        self.pipe= pipeline( 
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device,
        )
    def estimate_depth(self,image: Image.Image) -> np.ndarray:
        result = self.pipe(image)
        depth_pil=result["depth"]
        depth_raw=np.array(depth_pil, dtype=np.float32)
        min_val= depth_raw.min()
        max_val= depth_raw.max()
        depth_normalized= (depth_raw-min_val)/(max_val- min_val + 1e-8)
        depth_map=1.0-depth_normalized
        return depth_map