import argparse
import os
import cv2
import numpy as np
from PIL import Image

from src.depth import DepthEstimator
from src.segmentation import SceneSegmenter
from src.ordering import LayerOrderer
from src.inpainting import LayerInpainter
from src.intrinsics import IntrinsicDecomposer


def process_image(image_path: str, output_dir: str, device: str = None):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n1 Loading input image: {image_path}")
    raw_img = Image.open(image_path).convert("RGB")
    w, h = raw_img.size

    print("2 Running Depth Estimation (Depth-Anything-V2)")
    depth_estimator = DepthEstimator(device=device)
    depth_map = depth_estimator.estimate_depth(raw_img)
    
    depth_vis = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "depth_map.png"), depth_vis)

    print("3 Running Panoptic Segmentation (Mask2Former)")
    segmenter =SceneSegmenter(device=device)
    segments =segmenter.segment_scene(raw_img)
    print(f"      Identified {len(segments)} distinct scene segment(s).")

    print("4 Computing Occlusion Graph & Depth Sorting")
    orderer =LayerOrderer()
    ordered_layers = orderer.order_layers(segments, depth_map, (h, w))

    print("5 Performing Amodal Inpainting & Intrinsic Decomposition")
    inpainter =LayerInpainter()
    completed_layers = inpainter.process_layers(raw_img, ordered_layers)

    decomposer = IntrinsicDecomposer()

    print("\n--- Saving Layer Outputs ---")
    manifest = []
    for layer in completed_layers:
        idx = layer["order_index"]
        label = layer["label"].replace(" ", "_")
        depth_val = layer["depth"]
        rgba = layer["rgba"]

        base_name = f"layer_{idx:02d}_{label}"
        layer_path = os.path.join(output_dir, f"{base_name}.png")
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(layer_path, bgra)

        albedo_rgba, shading_rgba = decomposer.decompose(rgba)
        albedo_path = os.path.join(output_dir, f"{base_name}_albedo.png")
        shading_path = os.path.join(output_dir, f"{base_name}_shading.png")

        cv2.imwrite(albedo_path, cv2.cvtColor(albedo_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(shading_path, cv2.cvtColor(shading_rgba, cv2.COLOR_RGBA2BGRA))

        manifest.append(f"Layer {idx:02d} | Label: {label:<16} | Depth: {depth_val:.3f} | File: {base_name}.png")
        print(f" Saved: {base_name}.png (Depth: {depth_val:.3f}) + Albedo + Shading")

    manifest_path =os.path.join(output_dir, "layers_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest))

    print(f"\nPipeline successfully completed! All assets saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layered Scene Decomposition from a Single Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--output_dir", type=str, default="outputs/run", help="Target output folder")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    args = parser.parse_args()

    process_image(image_path=args.image, output_dir=args.output_dir, device=args.device)