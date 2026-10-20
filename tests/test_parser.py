"""
tests/test_parser.py
────────────────────
Unit tests for app/ocr/parser.py.

All Gemini API calls are mocked — no real network calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.request import SchemaField
from app.ocr.parser import ParserError, _clean_value, _normalise_date, parse

# ── Sample OCR text ───────────────────────────────────────────────────────────

SAMPLE_OCR_TEXT = """
FABRIC PURCHASE ORDER

Supplier: Reliance Ind. Ltd
Date: 15/06/2025
Invoice No: INV/2025/001

Item: Cotton Fabric
HSN Code: 5208
Quantity: 1,20,000 MTR
Rate Per Unit: ₹15.50
GST: 5%
Discount: 2%
Brokerage: 1.5%
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_schema(*fields: tuple[str, str]) -> dict:
    """Build a minimal output_schema dict for testing."""
    result = {}
    for name, ftype in fields:
        result[name] = SchemaField(
            type=ftype,
            master="supplier" if ftype == "auto_search" else None,
        )
    return result


def _mock_gemini_response(json_content: dict) -> MagicMock:
    """Build a mock Gemini generate_content response with JSON text."""
    response = MagicMock()
    response.text = json.dumps(json_content)
    return response


# ── Date normalisation ────────────────────────────────────────────────────────


class TestNormaliseDate:
    """Unit tests for _normalise_date helper."""

    def test_dd_slash_mm_slash_yyyy(self) -> None:
        assert _normalise_date("15/06/2025") == "2025-06-15"

    def test_dd_dash_mm_dash_yyyy(self) -> None:
        assert _normalise_date("01-01-2025") == "2025-01-01"

    def test_single_digit_day_month(self) -> None:
        assert _normalise_date("5/6/2025") == "2025-06-05"

    def test_already_iso_format_unchanged(self) -> None:
        # ISO format doesn't match DD/MM/YYYY pattern → returned as-is
        result = _normalise_date("2025-06-15")
        assert result == "2025-06-15"

    def test_non_date_string_unchanged(self) -> None:
        assert _normalise_date("NOT A DATE") == "NOT A DATE"


# ── Value cleaning ────────────────────────────────────────────────────────────


class TestCleanValue:
    """Unit tests for _clean_value helper."""

    def test_strips_rupee_symbol(self) -> None:
        assert _clean_value("rate", "₹15.50") == "15.50"

    def test_strips_rs_prefix(self) -> None:
        assert _clean_value("rate", "Rs. 250") == "250"

    def test_removes_lakh_commas(self) -> None:
        assert _clean_value("quantity", "1,20,000") == "120000"

    def test_normalises_date_field(self) -> None:
        assert _clean_value("invoice_date", "15/06/2025") == "2025-06-15"

    def test_none_returns_none(self) -> None:
        assert _clean_value("field", None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _clean_value("field", "") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _clean_value("field", "   ") is None

    def test_plain_string_unchanged(self) -> None:
        assert _clean_value("hsn_code", "5208") == "5208"

    def test_gst_percent_string_preserved(self) -> None:
        # "5%" doesn't match pure numeric pattern, so returned as-is
        result = _clean_value("gst_percent", "5%")
        assert result == "5%"

    def test_float_converted_to_2_decimal_places(self) -> None:
        assert _clean_value("quantity", "20.000") == "20.00"
        assert _clean_value("quantity", "6.667") == "6.67"
        assert _clean_value("quantity", "14.286") == "14.29"
        assert _clean_value("quantity", "88.189") == "88.19"

    def test_unwraps_dict_with_value_key(self) -> None:
        assert _clean_value("description_of_goods", {"value": "welvet : glossy : PT-3015 : Magenta Blue", "confidence": 0.95}) == "welvet : glossy : PT-3015 : Magenta Blue"

    def test_unwraps_dict_with_none_value(self) -> None:
        assert _clean_value("quality_master", {"value": None, "confidence": None}) is None

    def test_string_none_or_null_returns_none(self) -> None:
        assert _clean_value("field", "None") is None
        assert _clean_value("field", "null") is None


# ── Full parse flow ───────────────────────────────────────────────────────────


class TestParse:
    """Integration-style tests for parse() with mocked Gemini."""

    @pytest.mark.asyncio
    async def test_direct_field_extraction(self) -> None:
        """Direct fields should have their value and confidence extracted."""
        schema = _make_schema(("quantity", "direct"), ("hsn_code", "direct"))
        model_resp = _mock_gemini_response({
            "quantity": {"value": "1,20,000", "confidence": 0.96},
            "hsn_code": {"value": "5208", "confidence": 0.95},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

    @pytest.mark.asyncio
    async def test_line_item_supplier_qty_column_alias_fallback(self) -> None:
        """When Gemini returns qty/quantity/weight under alternate keys, it fallbacks into supplier_qty."""
        schema = {
            "line_items": SchemaField(
                type="line_items",
                item_fields={
                    "description_of_goods": {"label": "Description"},
                    "supplier_qty": {"label": "Qty"},
                    "primary_unit": {"label": "Unit"},
                },
            )
        }
        model_resp = _mock_gemini_response({
            "line_items": {
                "value": [
                    {"description_of_goods": "welvet", "qty": "20.000", "unit": "Kgs"},
                ],
                "confidence": 0.95,
            }
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)
            res = await parse("sample text", schema)

        items = res["line_items"]["value"]
        assert len(items) == 1
        assert items[0]["supplier_qty"] == "20.00"
        assert items[0]["primary_unit"] == "Kgs"

    @pytest.mark.asyncio
    async def test_direct_field_cleaning(self) -> None:
        schema = _make_schema(("quantity", "direct"), ("hsn_code", "direct"))
        model_resp = _mock_gemini_response({
            "quantity": {"value": "1,20,000", "confidence": 0.96},
            "hsn_code": {"value": "5208", "confidence": 0.95},
        })
        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)
            result = await parse("sample text", schema)

        assert result["quantity"]["value"] == "120000"   # commas stripped
        assert result["quantity"]["confidence"] == pytest.approx(0.96)
        assert result["hsn_code"]["value"] == "5208"
        assert result["hsn_code"]["confidence"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_null_returned_for_missing_field(self) -> None:
        """Fields not present in the document must return None, not a guess."""
        schema = _make_schema(("missing_field", "direct"))
        model_resp = _mock_gemini_response({
            "missing_field": {"value": None, "confidence": None},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert result["missing_field"]["value"] is None
        assert result["missing_field"]["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_indian_date_normalised(self) -> None:
        """Date fields in DD/MM/YYYY format must be normalised to YYYY-MM-DD."""
        schema = _make_schema(("invoice_date", "direct"))
        model_resp = _mock_gemini_response({
            "invoice_date": {"value": "15/06/2025", "confidence": 0.92},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert result["invoice_date"]["value"] == "2025-06-15"

    @pytest.mark.asyncio
    async def test_hsn_code_extraction(self) -> None:
        """HSN code should be returned as a plain numeric string."""
        schema = _make_schema(("hsn_code", "direct"))
        model_resp = _mock_gemini_response({
            "hsn_code": {"value": "5208", "confidence": 0.98},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert result["hsn_code"]["value"] == "5208"
        assert 0.90 <= result["hsn_code"]["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_within_expected_range(self) -> None:
        """All returned confidence values should be clamped to [0.0, 1.0]."""
        schema = _make_schema(("quantity", "direct"), ("rate", "direct"))
        model_resp = _mock_gemini_response({
            "quantity": {"value": "120", "confidence": 1.5},   # over-range
            "rate":     {"value": "15.50", "confidence": -0.1}, # under-range
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert 0.0 <= result["quantity"]["confidence"] <= 1.0
        assert 0.0 <= result["rate"]["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_auto_search_field_returns_raw_name(self) -> None:
        """auto_search fields should have entity name as 'value' (for matcher)."""
        schema = _make_schema(("supplier_name", "auto_search"))
        model_resp = _mock_gemini_response({
            "supplier_name": {"value": "Reliance Ind. Ltd", "confidence": 0.89},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert result["supplier_name"]["value"] == "Reliance Ind. Ltd"

    @pytest.mark.asyncio
    async def test_auto_populated_fields_skipped(self) -> None:
        """auto_populated fields must be entirely absent from parse() results."""
        from app.models.request import SchemaField

        schema = {
            "quantity":         SchemaField(type="direct"),
            "supplier_gst_no":  SchemaField(type="auto_populated"),
        }
        model_resp = _mock_gemini_response({
            "quantity": {"value": "500", "confidence": 0.95},
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        assert "quantity" in result
        assert "supplier_gst_no" not in result

    @pytest.mark.asyncio
    async def test_invalid_json_raises_parser_error(self) -> None:
        """Malformed JSON from Gemini should raise ParserError after retries."""
        bad_response = MagicMock()
        bad_response.text = "not valid json {"

        schema = _make_schema(("quantity", "direct"))

        with patch("app.ocr.parser.genai.Client") as MockClient, \
             patch("app.ocr.parser.asyncio.sleep", new_callable=AsyncMock):
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=bad_response)

            with pytest.raises(ParserError, match="invalid JSON"):
                await parse(SAMPLE_OCR_TEXT, schema)

    @pytest.mark.asyncio
    async def test_line_items_summary_row_filtered(self) -> None:
        """Total/summary rows in line_items must be stripped out automatically."""
        from app.models.request import SchemaField

        schema = {
            "items": SchemaField(
                type="line_items",
                item_fields={
                    "description_of_goods": "Item description",
                    "supplier_qty": "Quantity",
                },
            )
        }
        model_resp = _mock_gemini_response({
            "items": {
                "value": [
                    {
                        "description_of_goods": "welvet : glossy : PT-3015",
                        "supplier_qty": "80.000",
                    },
                    {
                        "description_of_goods": "muslin : mattic : YM-2926",
                        "supplier_qty": "80.000",
                    },
                    {
                        "description_of_goods": "Total",
                        "supplier_qty": "160.000",
                    },
                    {
                        "description_of_goods": "Grand Total Amount",
                        "supplier_qty": None,
                    },
                ],
                "confidence": 0.95,
            }
        })

        with patch("app.ocr.parser.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.aio.models.generate_content = AsyncMock(return_value=model_resp)

            result = await parse(SAMPLE_OCR_TEXT, schema)

        extracted_items = result["items"]["value"]
        assert len(extracted_items) == 2
        assert extracted_items[0]["description_of_goods"] == "welvet : glossy : PT-3015"
        assert extracted_items[1]["description_of_goods"] == "muslin : mattic : YM-2926"


class TestSummaryRowDetection:
    """Unit tests for _is_summary_or_header_row helper."""

    def test_detects_total_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": "Total", "qty": "160.000"}
        assert _is_summary_or_header_row(row) is True

    def test_detects_grand_total_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": "Grand Total Amount", "qty": "500"}
        assert _is_summary_or_header_row(row) is True

    def test_detects_subtotal_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": "Sub Total", "qty": "100"}
        assert _is_summary_or_header_row(row) is True

    def test_detects_total_qty_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": "Total Qty", "qty": "8"}
        assert _is_summary_or_header_row(row) is True

    def test_allows_valid_item_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": "welvet : glossy : PT-3015", "qty": "80.000"}
        assert _is_summary_or_header_row(row) is False

    def test_detects_empty_row(self) -> None:
        from app.ocr.parser import _is_summary_or_header_row
        row = {"description": None, "qty": "  "}
        assert _is_summary_or_header_row(row) is True

