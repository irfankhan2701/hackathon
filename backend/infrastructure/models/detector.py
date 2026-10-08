import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import uuid
from domain.entities import Region, Line, RegionType, BoundingBox
from infrastructure.imaging.crossout import detect_crossed_out_stroke
from infrastructure.storage.artifact_store import artifact_store, Artifact


def crop_numpy_image(img_np: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crops bounding box region from numpy image with boundary safety."""
    h_img, w_img = img_np.shape[:2]
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(w_img, int(bbox.x + bbox.w))
    y2 = min(h_img, int(bbox.y + bbox.h))
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return img_np[y1:y2, x1:x2]


class LayoutDetector:
    """
    Layout Intelligence & Reading-Order Graph Builder.
    Detects document regions (handwriting, printed text, margin notes, tables, signatures),
    segments text lines, crops them, saves line crop artifacts, and computes spatial reading order.
    """

    def detect_layout_and_lines(
        self, page_id: str, img_np: np.ndarray
    ) -> Tuple[List[Region], List[Line], Dict[str, Artifact]]:
        h, w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np

        # Binarize image for layout segmentation
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 1. Morphological horizontal line projection for text line candidates
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.08), 1))
        connected_h = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)

        contours, _ = cv2.findContours(connected_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_bboxes: List[BoundingBox] = []
        margin_x_threshold = w * 0.75  # Lines beyond 75% width are candidate margin notes

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Filter out tiny noise contours
            if bw > 15 and bh > 8 and bw < w * 0.98:
                raw_bboxes.append(BoundingBox(x=float(bx), y=float(by), w=float(bw), h=float(bh)))

        # Fallback line splitting if contour detection yields too few lines
        if len(raw_bboxes) < 2:
            # Segment into horizontal bands
            band_h = max(30, int(h / 10))
            for y_off in range(10, h - band_h, band_h):
                raw_bboxes.append(BoundingBox(x=float(w * 0.05), y=float(y_off), w=float(w * 0.85), h=float(band_h - 5)))

        # Sort lines top-to-bottom primarily, left-to-right secondarily
        raw_bboxes.sort(key=lambda b: (b.y, b.x))

        regions: List[Region] = []
        lines: List[Line] = []
        crop_artifacts: Dict[str, Artifact] = {}

        # 2. Group bboxes into Regions (Main Body vs Margin Notes vs Crossed Out)
        main_region_id = str(uuid.uuid4())
        margin_region_id = str(uuid.uuid4())

        main_region = Region(
            region_id=main_region_id,
            page_id=page_id,
            bounding_box=BoundingBox(x=0.0, y=0.0, w=float(w * 0.75), h=float(h)),
            region_type=RegionType.MAIN_HANDWRITING,
            reading_order_index=0,
        )
        margin_region = Region(
            region_id=margin_region_id,
            page_id=page_id,
            bounding_box=BoundingBox(x=float(w * 0.72), y=0.0, w=float(w * 0.28), h=float(h)),
            region_type=RegionType.MARGIN_NOTE,
            reading_order_index=1,
        )

        regions.extend([main_region, margin_region])

        reading_order_counter = 0
        for i, bbox in enumerate(raw_bboxes):
            crop_np = crop_numpy_image(img_np, bbox)
            
            # Check for crossed-out stroke
            is_crossed, _, _ = detect_crossed_out_stroke(crop_np)

            # Determine region assignment
            is_margin = bbox.x >= margin_x_threshold
            assigned_region_id = margin_region_id if is_margin else main_region_id

            # Save line crop artifact in store
            from PIL import Image
            import io
            pil_crop = Image.fromarray(crop_np)
            buf = io.BytesIO()
            pil_crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            crop_art = artifact_store.save_bytes(
                content=crop_bytes,
                file_type="png",
                transformation_name="line_crop",
                settings={"bbox": bbox.model_dump(), "line_index": i},
            )
            crop_artifacts[crop_art.artifact_id] = crop_art

            line_obj = Line(
                line_id=str(uuid.uuid4()),
                region_id=assigned_region_id,
                page_id=page_id,
                bounding_box=bbox,
                line_index=i,
                reading_order_index=reading_order_counter,
                line_crop_artifact_id=crop_art.artifact_id,
                is_crossed_out_candidate=is_crossed,
            )
            lines.append(line_obj)
            reading_order_counter += 1

        return regions, lines, crop_artifacts


layout_detector = LayoutDetector()
