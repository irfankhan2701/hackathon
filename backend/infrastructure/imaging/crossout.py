import cv2
import numpy as np
from typing import Tuple


def detect_crossed_out_stroke(crop_np: np.ndarray) -> Tuple[bool, float, float]:
    """
    Analyzes an image crop for crossed-out text strokes using Hough line detection,
    diagonal/horizontal stroke analysis, and stroke-to-ink intersection density.
    Returns (is_crossed_out, strike_density, confidence_score).
    """
    if crop_np is None or crop_np.size == 0:
        return False, 0.0, 0.0

    if len(crop_np.shape) == 3:
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_np.copy()

    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return False, 0.0, 0.0

    # Binarize ink
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_pixels = cv2.countNonZero(binary)
    if ink_pixels == 0:
        return False, 0.0, 0.0

    # Edge detection for lines
    edges = cv2.Canny(gray, 50, 150)

    # Detect line segments crossing horizontal or diagonal angle
    min_line_len = max(10, int(w * 0.45))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=25, minLineLength=min_line_len, maxLineGap=10
    )

    strike_line_mask = np.zeros_like(gray)
    strike_line_count = 0
    total_stroke_len = 0.0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            # Crossed-out strokes are horizontal (-30 to +30 deg) or diagonal (30 to 70 deg crossing text)
            if angle <= 35.0 or (angle >= 35.0 and angle <= 75.0):
                line_len = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                if line_len >= w * 0.4:
                    cv2.line(strike_line_mask, (x1, y1), (x2, y2), 255, thickness=2)
                    strike_line_count += 1
                    total_stroke_len += line_len

    # Intersect strike lines with ink pixels
    intersection = cv2.bitwise_and(binary, strike_line_mask)
    intersection_pixels = cv2.countNonZero(intersection)

    strike_density = float(intersection_pixels / max(1.0, float(ink_pixels)))
    relative_stroke_length = float(total_stroke_len / max(1.0, float(w)))

    # Decision criteria: horizontal/diagonal line spanning > 45% of width with ink intersection
    is_crossed_out = strike_line_count > 0 and (strike_density > 0.12 or relative_stroke_length > 0.6)
    confidence_score = float(min(1.0, (strike_density * 0.5 + relative_stroke_length * 0.5)))

    return is_crossed_out, strike_density, confidence_score
