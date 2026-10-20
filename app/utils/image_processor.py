"""
app/utils/image_processor.py
─────────────────────────────
Pre-processes document images before sending them to Mistral OCR to maximize extraction accuracy.

Techniques applied:
1. EXIF Auto-Orientation Correction (rotates mobile camera photos upright).
2. Document Perspective Warp & Boundary Detection (crops background clutter and flattens paper).
3. Deskew Angle Alignment (straightens tilted text lines).
4. Auto-Contrast & Sharpness Enhancement (makes small font digits crisp).
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


def preprocess_image(file_bytes: bytes, mime_type: str) -> bytes:
    """
    Enhance document image before OCR:
      1. Fix EXIF orientation (mobile phone portrait/landscape fix).
      2. Perspective warp / deskew if tilted.
      3. Apply auto-contrast & sharpness enhancement for small text fonts.

    Args:
        file_bytes: Raw binary bytes of uploaded image.
        mime_type: File MIME type string (e.g. "image/jpeg", "image/png").

    Returns:
        bytes: Pre-processed image bytes (or original bytes if unprocessable / PDF).
    """
    if mime_type not in ("image/jpeg", "image/png"):
        return file_bytes

    try:
        # 1. Open with Pillow and fix EXIF rotation (ensures camera photos are upright)
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)

        # Convert RGBA / P mode to RGB if needed
        # 2. Convert to OpenCV BGR for vision processing
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 3. Deskew image using Hough line text angle alignment
        img_bgr = _deskew_image(img_bgr)

        # 4. Convert back to Pillow for contrast & sharpness enhancement
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(img_rgb)

        enhanced_pil = ImageOps.autocontrast(enhanced_pil, cutoff=1)
        enhancer = ImageEnhance.Sharpness(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.3)

        out_buffer = io.BytesIO()
        save_format = "PNG" if mime_type == "image/png" else "JPEG"
        enhanced_pil.save(out_buffer, format=save_format, quality=95)
        processed_bytes = out_buffer.getvalue()

        logger.info(
            "Image preprocessing complete: size=%.1f KB",
            len(processed_bytes) / 1024,
        )
        return processed_bytes

    except Exception as exc:
        logger.warning("EXIF orientation / deskew failed, using original file: %s", exc)
        return file_bytes


def _deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Deskew document image by estimating dominant text/table line angles
    using Hough Line Transform.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=100, minLineLength=int(min(w, h) * 0.1), maxLineGap=20)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Normalize angle to [-45, 45] range for near-horizontal lines
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    logger.info("Detected document tilt angle: %.2f°", median_angle)

    if 0.5 < abs(median_angle) <= 45.0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated

    return img


def _four_point_transform_document(img: np.ndarray) -> Optional[np.ndarray]:
    """Find document boundary contour and warp to 2D rectangle."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check if 4 vertices found and area is > 20% of image area
        if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.20):
            pts = approx.reshape(4, 2)
            rect = _order_points(pts)
            (tl, tr, br, bl) = rect

            # Compute width and height of new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 contour points clockwise starting from top-left:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    cx, cy = np.mean(pts, axis=0)

    # Compute polar angle of each corner relative to centroid
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    shifted_angles = (angles + (3 * np.pi / 4)) % (2 * np.pi)
    order = np.argsort(shifted_angles)

    rect[0] = pts[order[0]]  # top-left
    rect[1] = pts[order[1]]  # top-right
    rect[2] = pts[order[2]]  # bottom-right
    rect[3] = pts[order[3]]  # bottom-left
    return rect
