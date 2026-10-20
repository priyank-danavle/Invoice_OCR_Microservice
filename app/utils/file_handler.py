"""
app/utils/file_handler.py
─────────────────────────
File validation and base64 encoding utilities.

Supported MIME types: image/jpeg, image/png, application/pdf.
Files larger than MAX_FILE_SIZE_MB are rejected before any API call.
"""

from __future__ import annotations

import base64
import logging
from typing import Tuple

from fastapi import UploadFile

from app.config import settings

logger = logging.getLogger(__name__)

# ── Allowed MIME types ────────────────────────────────────────────────────────

ALLOWED_MIME_TYPES: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png":  "png",
    "application/pdf": "pdf",
}


# ── Custom exceptions ─────────────────────────────────────────────────────────

class FileTooLargeError(Exception):
    """Raised when the uploaded file exceeds MAX_FILE_SIZE_BYTES."""


class UnsupportedFileTypeError(Exception):
    """Raised when the MIME type is not in ALLOWED_MIME_TYPES."""


class EmptyFileError(Exception):
    """Raised when the uploaded file has zero bytes."""


# ── Public helpers ────────────────────────────────────────────────────────────

async def read_and_validate(upload: UploadFile) -> Tuple[bytes, str]:
    """
    Read all bytes from an UploadFile, validate type and size.

    Args:
        upload: FastAPI UploadFile from the multipart form.

    Returns:
        Tuple[bytes, str]: (file_bytes, mime_type)

    Raises:
        UnsupportedFileTypeError: If content_type is not in ALLOWED_MIME_TYPES.
        FileTooLargeError: If file size exceeds the configured limit.
        EmptyFileError: If the file contains zero bytes.
    """
    mime_type: str = (upload.content_type or "").lower().strip()

    if mime_type not in ALLOWED_MIME_TYPES:
        raise UnsupportedFileTypeError(
            f"File type '{mime_type}' is not supported. "
            f"Accepted types: {', '.join(ALLOWED_MIME_TYPES)}."
        )

    file_bytes: bytes = await upload.read()

    if len(file_bytes) == 0:
        raise EmptyFileError("The uploaded file is empty (0 bytes).")

    if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise FileTooLargeError(
            f"File size {size_mb:.2f} MB exceeds the {settings.MAX_FILE_SIZE_MB} MB limit."
        )

    logger.debug(
        "File validated: name=%s mime=%s size_kb=%.1f",
        upload.filename,
        mime_type,
        len(file_bytes) / 1024,
    )
    return file_bytes, mime_type


def to_base64_data_uri(file_bytes: bytes, mime_type: str) -> str:
    """
    Encode raw bytes as a base64 data URI suitable for Mistral API calls.

    Args:
        file_bytes: Raw file content.
        mime_type: MIME type string (e.g. "image/jpeg").

    Returns:
        str: Data URI string "data:<mime>;base64,<encoded_data>".
    """
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def file_size_kb(file_bytes: bytes) -> float:
    """
    Return file size in kilobytes, rounded to one decimal place.

    Args:
        file_bytes: Raw file content.

    Returns:
        float: Size in KB.
    """
    return round(len(file_bytes) / 1024, 1)
