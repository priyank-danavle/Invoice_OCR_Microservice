"""
app/ocr/extractor.py
────────────────────
Extracts text from images and PDFs using Gemini Flash vision capability.

Supported formats:
    image/jpeg, image/png
    application/pdf
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import settings
from app.utils.image_processor import preprocess_image

logger = logging.getLogger(__name__)

# ── Custom exceptions ─────────────────────────────────────────────────────────

class OCRExtractionError(Exception):
    """
    Raised when Gemini OCR returns empty content or fails
    after exhausting retries.
    """


_OCR_PROMPT = """Extract all text from this document image/PDF exactly as written, formatted in clean, well-structured Markdown.

TABLE AND LAYOUT GUIDELINES:
- Convert ALL tabular data, line items, itemized listings, dispatch tables, and grid structures into explicit GitHub Flavored Markdown (GFM) tables using standard pipe syntax (`| Col 1 | Col 2 |`) with proper header divider lines (`| --- | --- |`).
- Ensure each product or service row is represented as a separate row in the markdown table.
- Read document text in its correct upright orientation even if the image is rotated sideways or upside down.
- Maintain strict row alignment: match each row's Serial Number (Sr. No.), Roll No., and Pcs No. strictly to its corresponding Description of Goods, Quantity, and Unit on that exact same horizontal line.
- EXTRACT ONLY VISIBLE TEXT: Extract only the text that is physically visible in the document image. NEVER guess, infer, auto-fill, or complete text that is covered or obscured under magnets, tape, or objects. If only a portion of a text is visible (e.g. "Magenta Blue" or "Sunrise red"), extract ONLY that visible portion exactly as it appears. If a cell is completely covered or blank, output `-`.
- Separate table summary rows (such as "Total", "Grand Total", "Sub Total", "Net Amount", "Total Quantity") from individual item rows by clearly labeling them (e.g. `| **TOTAL** | ... |`) or placing them as distinct key-value pairs outside the item table.
- Format floating-point numbers (quantities, weights, rates, amounts) to 2 decimal places (e.g., 20.00, 6.67, 14.29, 88.19).
- Preserve all headers, columns, key-value labels, and integer values exactly as they appear in the original document.

Return ONLY the raw Markdown text with no explanations, no code block markers, and no extra preamble."""


# Gemini 3.5 Flash-Lite standard paid-tier pricing
GEMINI_INPUT_COST_PER_1M = 0.30
GEMINI_OUTPUT_COST_PER_1M = 2.50


def _calculate_gemini_cost(response):
    """Calculate estimated Gemini API cost from response token usage."""

    usage = response.usage_metadata

    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or 0

    input_cost = (
        input_tokens / 1_000_000
    ) * GEMINI_INPUT_COST_PER_1M

    output_cost = (
        output_tokens / 1_000_000
    ) * GEMINI_OUTPUT_COST_PER_1M

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }



async def _extract_single_file(file_bytes: bytes, mime_type: str) -> Tuple[str, int]:
    """Internal helper to run Gemini OCR on a single file (image or PDF)."""
    if mime_type in ("image/jpeg", "image/png"):
        file_bytes = preprocess_image(file_bytes, mime_type)

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    max_attempts = 3
    base_delay = 2.0
    last_exc: Exception | None = None

    part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    models_to_try = [
        settings.GEMINI_PARSE_MODEL,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    for attempt in range(1, max_attempts + 1):
        for idx, model_name in enumerate(unique_models):
            try:
                logger.debug(
                    "Gemini OCR call attempt %d/%d (model=%s, mime=%s, size_kb=%.1f).",
                    attempt, max_attempts, model_name, mime_type, len(file_bytes) / 1024,
                )
                start = time.monotonic()
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=[part, _OCR_PROMPT],
                )

                cost_info = _calculate_gemini_cost(response)

                logger.info(
                    "Gemini OCR usage: model=%s, input_tokens=%d, output_tokens=%d, "
                    "total_tokens=%d, estimated_cost=$%.8f",
                    model_name,
                    cost_info["input_tokens"],
                    cost_info["output_tokens"],
                    cost_info["total_tokens"],
                    cost_info["total_cost_usd"],
                )

                elapsed = time.monotonic() - start
                logger.debug("Gemini OCR responded in %.2f s.", elapsed)

                raw_text = (response.text or "").strip()
                if not raw_text:
                    raise OCRExtractionError(
                        "Gemini OCR processed the document but extracted no text."
                    )

                logger.info(
                    "OCR complete: 1 page/file, %d characters extracted.\n--- RAW OCR TEXT START ---\n%s\n--- RAW OCR TEXT END ---",
                    len(raw_text), raw_text,
                )
                return raw_text, 1

            except OCRExtractionError:
                raise

            except APIError as exc:
                code = getattr(exc, "code", getattr(exc, "status_code", 0))
                if code in (404, 429, 500, 502, 503, 504) and idx < len(unique_models) - 1:
                    logger.warning(
                        "Gemini OCR model '%s' returned HTTP %s. Retrying with fallback model '%s'...",
                        model_name, code, unique_models[idx + 1],
                    )
                    last_exc = exc
                    if code in (429, 503, 504):
                        await asyncio.sleep(1.0)
                    continue
                if code in (429, 500, 502, 503, 504):
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini OCR rate-limited/unavailable (code=%s). Retrying in %.1f s … (attempt %d/%d)",
                        code, delay, attempt, max_attempts,
                    )
                    last_exc = exc
                    await asyncio.sleep(delay)
                    break
                raise OCRExtractionError(f"Gemini OCR API error (code={code}): {exc}") from exc

            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Unexpected error during Gemini OCR extraction: %s", exc)
                raise OCRExtractionError(
                    f"Unexpected error during Gemini OCR extraction: {exc}"
                ) from exc

    raise OCRExtractionError(
        f"Gemini OCR extraction failed after {max_attempts} attempts. Last error: {last_exc}"
    )


async def extract(
    files: bytes | List[Tuple[bytes, str]],
    mime_type: str = "",
) -> Tuple[str, int]:
    """
    Run Gemini OCR on supplied file(s) and return the extracted text.

    Args:
        files: Either raw binary content of a single file (with mime_type set),
               or a list of (file_bytes, mime_type) tuples for multi-image/page processing.
        mime_type: MIME type string (e.g. "image/jpeg", "application/pdf") if files is bytes.

    Returns:
        Tuple[str, int]: (raw_text, page_count)
            raw_text    — Full markdown text from the document(s).
            page_count  — Total number of files/pages processed.

    Raises:
        OCRExtractionError: If Gemini returns empty content or fails.
    """
    if isinstance(files, bytes):
        return await _extract_single_file(files, mime_type)

    if not files:
        raise OCRExtractionError("No files provided for OCR extraction.")

    if len(files) == 1:
        return await _extract_single_file(files[0][0], files[0][1])

    # Run OCR on all files in parallel
    results = await asyncio.gather(
        *[_extract_single_file(fb, mt) for fb, mt in files]
    )

    combined_texts: List[str] = []
    total_pages = 0
    for idx, (raw_text, page_count) in enumerate(results):
        total_pages += page_count
        combined_texts.append(f"--- PAGE {idx + 1} ---\n{raw_text}")

    full_markdown = "\n\n".join(combined_texts)
    return full_markdown, total_pages
