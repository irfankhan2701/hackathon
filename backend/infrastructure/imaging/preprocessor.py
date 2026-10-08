import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Dict, List, Tuple, Optional
from domain.entities import Artifact, QualityMetrics
from infrastructure.storage.artifact_store import artifact_store
from infrastructure.imaging.quality import calculate_quality_metrics


def load_image_bytes(image_bytes: bytes, filename: str = "") -> List[np.ndarray]:
    """Loads raw image bytes (PNG, JPG, PDF) into numpy array RGB images."""
    lower_fn = filename.lower()
    images = []

    if lower_fn.endswith(".pdf") or image_bytes.startswith(b"%PDF"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(image_bytes))
            # Fallback if pdf2image or fitz available
            try:
                from pdf2image import convert_from_bytes
                pil_imgs = convert_from_bytes(image_bytes)
                for img in pil_imgs:
                    images.append(np.array(img))
            except Exception:
                # Render synthetic fallback page if pdf2image binary not present
                pass
        except Exception:
            pass

    if not images:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        images.append(np.array(pil_img.convert("RGB")))

    return images


def correct_perspective(img_np: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Detects page boundaries and applies homography perspective correction if found."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np, False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    page_contour = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img_np.shape[0] * img_np.shape[1] * 0.2):
            page_contour = approx
            break

    if page_contour is None:
        return img_np, False

    pts = page_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))
    return warped, True


def deskew_image(img_np: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image to fix skew angle."""
    if abs(angle) < 0.5:
        return img_np
    (h, w) = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """Applies CLAHE contrast enhancement."""
    if len(img_np.shape) == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(img_np)


def apply_adaptive_binarization(img_np: np.ndarray) -> np.ndarray:
    """Applies Sauvola/Adaptive thresholding."""
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)


def apply_denoising(img_np: np.ndarray) -> np.ndarray:
    """Applies non-local means denoising."""
    if len(img_np.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img_np, None, 7, 7, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(img_np, None, 7, 7, 21)


def apply_sharpening(img_np: np.ndarray) -> np.ndarray:
    """Applies mild unsharp masking."""
    gaussian = cv2.GaussianBlur(img_np, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_np, 1.5, gaussian, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def np_to_bytes(img_np: np.ndarray, ext: str = "png") -> bytes:
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format=ext.upper())
    return buf.getvalue()


def process_and_create_artifacts(
    raw_bytes: bytes, filename: str
) -> Tuple[Dict[str, Artifact], QualityMetrics, np.ndarray, int, int]:
    """
    Processes raw file bytes, generates all required immutable image variants,
    persists them into ArtifactStore, and returns variant map + quality metrics.
    """
    imgs = load_image_bytes(raw_bytes, filename)
    img_np = imgs[0] if imgs else np.zeros((800, 600, 3), dtype=np.uint8)
    h, w = img_np.shape[:2]

    # Save raw artifact
    raw_art = artifact_store.save_bytes(
        content=raw_bytes,
        file_type=filename.split(".")[-1] if "." in filename else "png",
        transformation_name="raw",
        settings={"filename": filename},
    )

    # 1. Perspective normalization
    norm_np, homography_applied = correct_perspective(img_np)
    norm_art = artifact_store.save_bytes(
        content=np_to_bytes(norm_np),
        file_type="png",
        transformation_name="normalized",
        settings={"homography_applied": homography_applied},
        source_artifact_id=raw_art.artifact_id,
    )

    # 2. Quality metrics & deskew
    metrics = calculate_quality_metrics(norm_np)
    deskewed_np = deskew_image(norm_np, metrics.skew_angle)
    deskewed_art = artifact_store.save_bytes(
        content=np_to_bytes(deskewed_np),
        file_type="png",
        transformation_name="deskewed",
        settings={"skew_angle": metrics.skew_angle},
        source_artifact_id=norm_art.artifact_id,
    )

    # 3. CLAHE
    clahe_np = apply_clahe(deskewed_np)
    clahe_art = artifact_store.save_bytes(
        content=np_to_bytes(clahe_np),
        file_type="png",
        transformation_name="clahe",
        settings={"clipLimit": 2.5, "tileGridSize": [8, 8]},
        source_artifact_id=deskewed_art.artifact_id,
    )

    # 4. Adaptive Binarized
    bin_np = apply_adaptive_binarization(deskewed_np)
    bin_art = artifact_store.save_bytes(
        content=np_to_bytes(bin_np),
        file_type="png",
        transformation_name="adaptive_binarized",
        settings={"method": "adaptive_gaussian", "blockSize": 21, "C": 10},
        source_artifact_id=deskewed_art.artifact_id,
    )

    # 5. Denoised
    denoised_np = apply_denoising(deskewed_np)
    denoised_art = artifact_store.save_bytes(
        content=np_to_bytes(denoised_np),
        file_type="png",
        transformation_name="denoised",
        settings={"h": 7, "templateWindowSize": 7, "searchWindowSize": 21},
        source_artifact_id=deskewed_art.artifact_id,
    )

    # 6. Sharpened (if blur score is low)
    if metrics.blur_score < 150:
        sharp_np = apply_sharpening(deskewed_np)
        sharp_art = artifact_store.save_bytes(
            content=np_to_bytes(sharp_np),
            file_type="png",
            transformation_name="sharpened",
            settings={"weight": 1.5},
            source_artifact_id=deskewed_art.artifact_id,
        )
    else:
        sharp_art = deskewed_art

    artifacts = {
        "raw": raw_art,
        "normalized": norm_art,
        "deskewed": deskewed_art,
        "clahe": clahe_art,
        "adaptive_binarized": bin_art,
        "denoised": denoised_art,
        "sharpened": sharp_art,
    }

    return artifacts, metrics, deskewed_np, w, h
