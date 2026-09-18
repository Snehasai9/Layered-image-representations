from typing import Dict, List
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class SceneSegmenter:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model_id="facebook/mask2former-swin-tiny-coco-panoptic"
        self.processor=AutoImageProcessor.from_pretrained(self.model_id)
        self.model=Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()

    def segment_scene(self, image: Image.Image) -> List[Dict]:
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs=self.model(**inputs)
            target_sizes = [(image.height, image.width)]        
            results=(
            self.processor.post_process_panoptic_segmentation(
                outputs, target_sizes=target_sizes)[0]
        )
        segmentation_map=results["segmentation"].cpu().numpy()
        segments_info=results["segments_info"]

        layers = []
        for segment in segments_info:
            seg_id=segment["id"]
            label_id=segment["label_id"]
            label_name=self.model.config.id2label.get(
                label_id,f"class_{label_id}"
            )
            binary_mask=segmentation_map == seg_id
            if np.sum(binary_mask) < 100:
                continue
            layers.append(
                {"id": seg_id, "label": label_name, "mask": binary_mask}
            )
        return layers
        