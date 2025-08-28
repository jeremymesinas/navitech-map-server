import cv2 as cv
import numpy as np
from typing import Optional
from ultralytics import YOLO
from app.svg_utils import contours_to_svg


def process_yolo_to_svg(image_path: str, model: Optional[YOLO]) -> str:
    if model is None:
        raise RuntimeError("Model not loaded yet.")

    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]

    # Inference (segmentation model)
    results = model(img)
    masks = results[0].masks  # None if no segmentations
    if masks is None or masks.data is None or len(masks.data) == 0:
        # Return an empty but valid SVG with the original size
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"></svg>'

    # Convert each mask to contours
    paths = []
    for mask in masks.data:  # [N, H, W]
        m = (mask.cpu().numpy().astype(np.uint8)) * 255
        # Clean edges a bit (optional)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
        # Find outer contours
        contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Simplify contours with approxPolyDP to reduce points (~0.5% perimeter)
        simplified = []
        for cnt in contours:
            if cv.contourArea(cnt) < 50:  # ignore tiny artifacts
                continue
            eps = 0.005 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, eps, True)
            simplified.append(approx)
        if simplified:
            paths.extend(simplified)

    # Convert all contours to a single SVG string
    svg = contours_to_svg(paths, width=w, height=h)
    return svg
