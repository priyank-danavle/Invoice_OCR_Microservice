"""
tests/test_extractor.py
───────────────────────
Unit tests for app/ocr/extractor.py.

All Gemini API calls are mocked — no real network calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ocr.extractor import OCRExtractionError, extract


def _make_gemini_ocr_response(text: str) -> MagicMock:
    """Build a mock Gemini response object."""
    resp = MagicMock()
    resp.text = text
    return resp


def _fake_jpg_bytes() -> bytes:
    """Minimal 1×1 JPEG-like bytes for testing."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _fake_pdf_bytes() -> bytes:
    """Minimal PDF-like bytes for testing."""
    return b"%PDF-1.4 " + b"\x00" * 100


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestExtractJPG:
    """OCR extraction from JPEG images."""

    @pytest.mark.asyncio
    async def test_jpg_returns_nonempty_text(self) -> None:
        """A valid JPEG should produce non-empty raw text."""
        mock_response = _make_gemini_ocr_response("Invoice\nSupplier: Reliance Industries\nAmount: 50000")

        with patch("app.ocr.extractor.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

            raw_text, page_count = await extract(_fake_jpg_bytes(), "image/jpeg")

        assert raw_text != ""
        assert "Invoice" in raw_text
        assert page_count == 1

    @pytest.mark.asyncio
    async def test_png_returns_nonempty_text(self) -> None:
        """A valid PNG should produce non-empty raw text."""
        mock_response = _make_gemini_ocr_response("PNG Invoice Content")

        with patch("app.ocr.extractor.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

            raw_text, page_count = await extract(_fake_jpg_bytes(), "image/png")

        assert raw_text != ""
        assert page_count == 1

    @pytest.mark.asyncio
    async def test_multiple_images_extraction(self) -> None:
        """Passing multiple images should return combined text and correct page count."""
        mock_response = _make_gemini_ocr_response("Page 1 Content\nPage 2 Content")

        with patch("app.ocr.extractor.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

            files = [(_fake_jpg_bytes(), "image/jpeg"), (_fake_jpg_bytes(), "image/jpeg")]
            raw_text, page_count = await extract(files)

        assert "Page 1 Content" in raw_text
        assert page_count == 2


class TestExtractPDF:
    """OCR extraction from PDF documents."""

    @pytest.mark.asyncio
    async def test_pdf_returns_nonempty_text(self) -> None:
        """A valid PDF should produce non-empty raw text."""
        mock_response = _make_gemini_ocr_response("PDF Invoice\nHSN: 5208\nQuantity: 120 MTR")

        with patch("app.ocr.extractor.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

            raw_text, page_count = await extract(_fake_pdf_bytes(), "application/pdf")

        assert "HSN" in raw_text
        assert page_count == 1


class TestExtractErrors:
    """Error handling in the extractor."""

    @pytest.mark.asyncio
    async def test_empty_text_raises_ocr_error(self) -> None:
        """Empty OCR text response should raise OCRExtractionError."""
        mock_response = _make_gemini_ocr_response("   ")

        with patch("app.ocr.extractor.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

            with pytest.raises(OCRExtractionError, match="extracted no text"):
                await extract(_fake_jpg_bytes(), "image/jpeg")


class TestImageProcessor:
    """Image preprocessing unit tests."""

    def test_preprocess_image_jpeg(self) -> None:
        """Preprocessing a JPEG image should return valid processed bytes."""
        from PIL import Image
        import io
        from app.utils.image_processor import preprocess_image

        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        processed = preprocess_image(raw_bytes, "image/jpeg")
        assert processed is not None
        assert len(processed) > 0

    def test_preprocess_pdf_returns_original_bytes(self) -> None:
        """PDF files should bypass image preprocessing and return original bytes."""
        from app.utils.image_processor import preprocess_image
        pdf_bytes = b"%PDF-1.4 sample content"
        res = preprocess_image(pdf_bytes, "application/pdf")
        assert res == pdf_bytes
