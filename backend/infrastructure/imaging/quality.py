import cv2
import numpy as np
from domain.entities import QualityMetrics, QualityClass


def calculate_quality_metrics(image_np: np.ndarray) -> QualityMetrics:
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    height, width = gray.shape[:2]

    # Blur score using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = float(round(laplacian_var, 2))

    # Contrast score using standard deviation
    contrast_score = float(round(np.std(gray), 2))

    # Skew angle estimation using Hough lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=width * 0.3, maxLineGap=20)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 45.0:
                angles.append(angle)
    
    skew_angle = float(round(np.median(angles), 2)) if len(angles) > 0 else 0.0

    # High frequency noise score ratio
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    noise_score = float(round(np.mean(grad_mag) / max(1.0, float(np.mean(gray))), 2))

    # Assign Quality Class based on calibrated thresholds
    if blur_score > 300 and contrast_score > 45 and noise_score < 3.0:
        quality_class = QualityClass.EXCELLENT
    elif blur_score > 150 and contrast_score > 30:
        quality_class = QualityClass.GOOD
    elif blur_score > 50 and contrast_score > 15:
        quality_class = QualityClass.FAIR
    else:
        quality_class = QualityClass.POOR

    return QualityMetrics(
        blur_score=blur_score,
        contrast_score=contrast_score,
        skew_angle=skew_angle,
        resolution=[width, height],
        noise_score=noise_score,
        quality_class=quality_class,
    )
