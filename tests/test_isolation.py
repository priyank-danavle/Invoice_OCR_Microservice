"""
tests/test_isolation.py
───────────────────────
Multi-client isolation tests for the OCR Invoice Reader microservice.

These tests verify that:
  1. Client A cannot access Client B's entity records.
  2. Requests without client_id are rejected with HTTP 400.
  3. Requests with an unknown client_id are rejected with HTTP 400.

The FastAPI test client is used; Mistral and MySQL are fully mocked.
"""

from __future__ import annotations

import json
import io
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

# ── Test client setup ─────────────────────────────────────────────────────────

API_KEY = "test-api-key-for-isolation"

# ── Index fixture data ────────────────────────────────────────────────────────

CLIENT_1_SUPPLIERS: List[Dict[str, Any]] = [
    {"id": 1, "name": "Client1 Supplier Alpha", "client_id": 1},
    {"id": 2, "name": "Client1 Supplier Beta",  "client_id": 1},
]

CLIENT_2_SUPPLIERS: List[Dict[str, Any]] = [
    {"id": 100, "name": "Client2 Supplier Gamma", "client_id": 2},
    {"id": 101, "name": "Client2 Supplier Delta", "client_id": 2},
]

SAMPLE_OUTPUT_SCHEMA = json.dumps({
    "supplier_name": {"type": "auto_search", "master": "supplier"},
    "quantity":      {"type": "direct"},
})

# ── Shared mock helpers ───────────────────────────────────────────────────────

def _make_ocr_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp



def _make_parse_response(fields: Dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(fields)
    return resp



def _fake_jpg() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _upload_file():
    return ("file", ("invoice.jpg", io.BytesIO(_fake_jpg()), "image/jpeg"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_api_key(monkeypatch):
    """Override MICROSERVICE_API_KEY so tests don't need a real secret."""
    monkeypatch.setattr(settings, "MICROSERVICE_API_KEY", API_KEY)


@pytest.fixture()
def client() -> TestClient:
    """Return a FastAPI TestClient (lifespan startup/shutdown mocked)."""
    with patch("app.main.indexer.load", new_callable=AsyncMock), \
         patch("app.main.indexer.start_background_refresh"), \
         patch("app.main.indexer.stop_background_refresh"), \
         patch("app.main.indexer.close_pool", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestClientIsolation:
    """Verify entity index isolation between different client_ids."""

    def test_client_a_cannot_see_client_b_records(self, client: TestClient) -> None:
        """
        When client_id=1 makes a request, only client 1's entity records
        should be searched — client 2's records must never appear.
        """
        captured_client_ids: List[int] = []

        def mock_get_client_index(master_key: str, client_id: int):
            captured_client_ids.append(client_id)
            if client_id == 1:
                return CLIENT_1_SUPPLIERS
            raise Exception(f"Unexpected client_id={client_id} accessed!")

        ocr_resp   = _make_ocr_response("Supplier: Client1 Supplier Alpha\nQty: 100")
        parse_resp = _make_parse_response({
            "supplier_name": {"value": "Client1 Supplier Alpha", "confidence": 0.95},
            "quantity":      {"value": "100", "confidence": 0.97},
        })

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.ocr.extractor.genai.Client") as MockOCR, \
             patch("app.ocr.parser.genai.Client") as MockParse, \
             patch("app.ocr.matcher.get_client_index", side_effect=mock_get_client_index):

            MockOCR.return_value.aio.models.generate_content = AsyncMock(return_value=ocr_resp)
            MockParse.return_value.aio.models.generate_content = AsyncMock(return_value=parse_resp)

            response = client.post(
                "/api/v1/invoice/parse",
                headers={"X-API-Key": API_KEY},
                files=[_upload_file()],
                data={"client_id": "1", "output_schema": SAMPLE_OUTPUT_SCHEMA},
            )

        assert response.status_code == 200, response.text

        # All index accesses must be scoped to client_id=1
        for accessed_id in captured_client_ids:
            assert accessed_id == 1, (
                f"Expected only client_id=1 to be accessed, but got {accessed_id}"
            )

        # Verify no client-2 entities appear in the response
        data = response.json()
        data_section = data.get("data") or data.get("basic_details") or {}
        supplier_result = data_section.get("supplier_name", {})
        matched_label = supplier_result.get("matched_label", "")
        assert "Client2" not in (matched_label or ""), (
            f"Client 2's data leaked into client 1's response: {matched_label}"
        )

    def test_client_b_cannot_see_client_a_records(self, client: TestClient) -> None:
        """
        When client_id=2 makes a request, only client 2's records are used.
        """
        captured_client_ids: List[int] = []

        def mock_get_client_index(master_key: str, client_id: int):
            captured_client_ids.append(client_id)
            if client_id == 2:
                return CLIENT_2_SUPPLIERS
            raise Exception(f"Unexpected client_id={client_id} accessed!")

        ocr_resp   = _make_ocr_response("Supplier: Client2 Supplier Gamma\nQty: 50")
        parse_resp = _make_parse_response({
            "supplier_name": {"value": "Client2 Supplier Gamma", "confidence": 0.93},
            "quantity":      {"value": "50", "confidence": 0.95},
        })

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.ocr.extractor.genai.Client") as MockOCR, \
             patch("app.ocr.parser.genai.Client") as MockParse, \
             patch("app.ocr.matcher.get_client_index", side_effect=mock_get_client_index):

            MockOCR.return_value.aio.models.generate_content = AsyncMock(return_value=ocr_resp)
            MockParse.return_value.aio.models.generate_content = AsyncMock(return_value=parse_resp)

            response = client.post(
                "/api/v1/invoice/parse",
                headers={"X-API-Key": API_KEY},
                files=[_upload_file()],
                data={"client_id": "2", "output_schema": SAMPLE_OUTPUT_SCHEMA},
            )

        assert response.status_code == 200, response.text

        for accessed_id in captured_client_ids:
            assert accessed_id == 2, (
                f"Expected only client_id=2, but got {accessed_id}"
            )


class TestMissingClientId:
    """Requests without client_id must be rejected at HTTP 400."""

    def test_missing_client_id_returns_400(self, client: TestClient) -> None:
        """
        Omitting client_id entirely from the form data must produce HTTP 422
        (FastAPI form validation) or HTTP 400 — not 200.
        """
        response = client.post(
            "/api/v1/invoice/parse",
            headers={"X-API-Key": API_KEY},
            files=[_upload_file()],
            data={"output_schema": SAMPLE_OUTPUT_SCHEMA},
            # client_id deliberately omitted
        )
        # FastAPI returns 422 for missing required Form fields
        assert response.status_code in (400, 422), (
            f"Expected 400 or 422 for missing client_id, got {response.status_code}"
        )

    def test_zero_client_id_returns_400(self, client: TestClient) -> None:
        """
        client_id=0 is invalid (must be > 0) and must be rejected HTTP 400.
        """
        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.ocr.extractor.genai.Client"):

            response = client.post(
                "/api/v1/invoice/parse",
                headers={"X-API-Key": API_KEY},
                files=[_upload_file()],
                data={"client_id": "0", "output_schema": SAMPLE_OUTPUT_SCHEMA},
            )

        assert response.status_code == 400, response.text
        assert "client_id" in response.text.lower()

    def test_negative_client_id_returns_400(self, client: TestClient) -> None:
        """Negative client_id values must be rejected with HTTP 400."""
        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.ocr.extractor.genai.Client"):

            response = client.post(
                "/api/v1/invoice/parse",
                headers={"X-API-Key": API_KEY},
                files=[_upload_file()],
                data={"client_id": "-5", "output_schema": SAMPLE_OUTPUT_SCHEMA},
            )

        assert response.status_code == 400, response.text


class TestInvalidClientId:
    """Requests with a client_id not present in the entity index must be rejected."""

    def test_unknown_client_id_returns_400(self, client: TestClient) -> None:
        """
        A client_id with no records in the entity index must trigger HTTP 400,
        not a 500 or a silent empty match.
        """
        from app.db.indexer import ClientNotFoundError

        ocr_resp   = _make_ocr_response("Supplier: Some Supplier\nQty: 10")
        parse_resp = _make_parse_response({
            "supplier_name": {"value": "Some Supplier", "confidence": 0.90},
            "quantity":      {"value": "10", "confidence": 0.95},
        })

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.ocr.extractor.genai.Client") as MockOCR, \
             patch("app.ocr.parser.genai.Client") as MockParse, \
             patch(
                 "app.ocr.matcher.get_client_index",
                 side_effect=ClientNotFoundError("client_id=9999 not found"),
             ):
            MockOCR.return_value.aio.models.generate_content = AsyncMock(return_value=ocr_resp)
            MockParse.return_value.aio.models.generate_content = AsyncMock(return_value=parse_resp)

            response = client.post(
                "/api/v1/invoice/parse",
                headers={"X-API-Key": API_KEY},
                files=[_upload_file()],
                data={"client_id": "9999", "output_schema": SAMPLE_OUTPUT_SCHEMA},
            )

        assert response.status_code == 400, response.text
        error_body = response.json()
        assert "9999" in error_body.get("message", "") or \
               "client" in error_body.get("message", "").lower()


class TestAuthIsolation:
    """Verify that missing or wrong API keys are rejected for all endpoints."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        """No X-API-Key header → HTTP 401."""
        response = client.post(
            "/api/v1/invoice/parse",
            files=[_upload_file()],
            data={"client_id": "1", "output_schema": SAMPLE_OUTPUT_SCHEMA},
        )
        assert response.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        """Incorrect X-API-Key → HTTP 401."""
        response = client.post(
            "/api/v1/invoice/parse",
            headers={"X-API-Key": "totally-wrong-key"},
            files=[_upload_file()],
            data={"client_id": "1", "output_schema": SAMPLE_OUTPUT_SCHEMA},
        )
        assert response.status_code == 401

    def test_health_check_no_auth_required(self, client: TestClient) -> None:
        """GET /health must be publicly accessible without an API key."""
        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.indexer.indexed_client_count", return_value=5):
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProcessNonBlockingExactMatch:
    """Verify non-blocking exact match behavior in /api/v1/invoice/process."""

    def test_top_level_unmatched_auto_search_passes_raw_name_and_warning(self, client: TestClient) -> None:
        """Unmatched top-level auto_search field should pass the raw unmatched field name in basic_details and return 200 with warning_message."""
        from app.ocr.matcher import EntityMatchError

        schema = json.dumps({
            "supplier_name": {"type": "auto_search", "master": "supplier"},
            "invoice_no":    {"type": "direct"},
        })
        extracted_data = {
            "supplier_name": {"value": "NonExistentSupplier", "confidence": 0.9},
            "invoice_no":    {"value": "INV-100", "confidence": 0.95},
        }

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=EntityMatchError("No exact match found for supplier")):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["basic_details"]["supplier_name"] == "NonExistentSupplier"
        assert body["basic_details"]["invoice_no"] == "INV-100"
        assert body["warning_message"] is not None
        assert "Supplier Name ('NonExistentSupplier') not found" in body["warning_message"]

    def test_invalid_line_item_master_removes_line_item_and_adds_warning(self, client: TestClient) -> None:
        """Line items with invalid quality/design/chart/color master are removed from line_items."""
        from app.ocr.matcher import EntityMatchError, MatchResult

        schema = json.dumps({
            "items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                    "supplier_qty": "Qty",
                },
            }
        })
        extracted_data = {
            "items": {
                "value": [
                    {
                        "description_of_goods": "Valid Quality : Design A",
                        "quality_master": "Valid Quality",
                        "supplier_qty": "80.000",
                    },
                    {
                        "description_of_goods": "Invalid Quality : Design B",
                        "quality_master": "BadQuality",
                        "supplier_qty": "50.000",
                    },
                ],
                "confidence": 0.95,
            }
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            if raw == "BadQuality":
                raise EntityMatchError("No exact match")
            return MatchResult(raw=raw, matched_id=10, matched_label=raw, confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        # Only row 1 remains (row 2 with BadQuality was removed)
        assert len(body["line_items"]) == 1
        assert body["line_items"][0]["quality_master"]["label"] == "Valid Quality"
        assert body["warning_message"] is not None
        assert "Row 2" in body["warning_message"]
        assert "BadQuality" in body["warning_message"]

    def test_empty_master_field_removes_line_item_row(self, client) -> None:
        """When quality_master is empty string '', the row is removed without splitting description_of_goods."""
        from app.ocr.matcher import EntityMatchError, MatchResult

        schema = json.dumps({
            "supplier_name": {"type": "direct", "section": "basic_details"},
            "line_items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                },
            },
        })

        extracted_data = {
            "supplier_name": "WELKNIT",
            "line_items": [
                {
                    "description_of_goods": "welvet : glossy : PT-3015 : Magenta Blue",
                    "quality_master": "",
                },
            ],
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            if not raw:
                raise EntityMatchError("Field value is empty")
            return MatchResult(raw=raw, matched_id=10, matched_label=raw, confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["line_items"]) == 0
        assert body["warning_message"] is not None
        assert "Row 1 removed from line items due to invalid Quality Master" in body["warning_message"]

    def test_zero_or_empty_quantity_removes_line_item_row(self, client) -> None:
        """Rows with quantity null, 0, '0', '0.00', or '' are removed from line items."""
        from app.ocr.matcher import MatchResult

        schema = json.dumps({
            "line_items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                    "qty": "Qty",
                },
            },
        })

        extracted_data = {
            "line_items": [
                {"description_of_goods": "Item 1", "quality_master": "welvet", "qty": "20.00"},
                {"description_of_goods": "Item 2", "quality_master": "welvet", "qty": "0"},
                {"description_of_goods": "Item 3", "quality_master": "welvet", "qty": None},
                {"description_of_goods": "Item 4", "quality_master": "welvet", "qty": "0.00"},
            ],
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            return MatchResult(raw=raw, matched_id=10, matched_label=raw, confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["line_items"]) == 1
        assert body["line_items"][0]["description_of_goods"] == "Item 1"
        assert body["warning_message"] is not None
        assert "Row 2 removed" in body["warning_message"]
        assert "Row 3 removed" in body["warning_message"]
        assert "Row 4 removed" in body["warning_message"]

    def test_invalid_alt_unit_retains_line_item_row(self, client) -> None:
        """Line items with an unmapped/invalid alt_unit are preserved rather than removed."""
        from app.ocr.matcher import EntityMatchError, MatchResult

        schema = json.dumps({
            "line_items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                    "alt_unit": {"master": "unit"},
                    "qty": "Qty",
                },
            },
        })

        extracted_data = {
            "line_items": [
                {"description_of_goods": "Item 1", "quality_master": "welvet", "alt_unit": "UnknownUnit", "qty": "20.00"},
            ],
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            if master_key == "unit" and raw == "UnknownUnit":
                raise EntityMatchError("No exact match found for unit 'UnknownUnit'")
            return MatchResult(raw=raw, matched_id=10, matched_label=raw, confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["line_items"]) == 1
        row = body["line_items"][0]
        assert row["description_of_goods"] == "Item 1"
        assert row["alt_unit"] == {"id": None, "label": "UnknownUnit"}
        assert body["warning_message"] is not None
        assert "Alt Unit ('UnknownUnit') not found" in body["warning_message"]

    def test_empty_or_null_alt_unit_does_not_raise_validation(self, client) -> None:
        """When alt_unit is empty string or None, no validation warning is raised for alt_unit."""
        from app.ocr.matcher import MatchResult

        schema = json.dumps({
            "line_items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                    "alt_unit": {"master": "unit"},
                    "qty": "Qty",
                },
            },
        })

        extracted_data = {
            "line_items": [
                {"description_of_goods": "Item 1", "quality_master": "welvet", "alt_unit": "", "qty": "20.00"},
                {"description_of_goods": "Item 2", "quality_master": "welvet", "alt_unit": None, "qty": "15.00"},
            ],
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            return MatchResult(raw=raw, matched_id=10, matched_label=raw, confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["line_items"]) == 2
        assert body.get("warning_message") is None

    def test_valid_alt_unit_with_zero_or_negative_alt_qty_raises_validation_without_popping_row(self, client) -> None:
        """When alt_unit is valid matched unit, but alt_qty is <= 0 or empty, a validation warning is raised without popping the row."""
        from app.ocr.matcher import MatchResult

        schema = json.dumps({
            "line_items": {
                "type": "line_items",
                "item_fields": {
                    "description_of_goods": "Description",
                    "quality_master": {"master": "quality"},
                    "alt_unit": {"master": "unit"},
                    "alt_qty": "Alt Qty",
                    "qty": "Qty",
                },
            },
        })

        extracted_data = {
            "line_items": [
                {"description_of_goods": "Item 1", "quality_master": "welvet", "alt_unit": "Mtrs", "alt_qty": "0", "qty": "20.00"},
                {"description_of_goods": "Item 2", "quality_master": "welvet", "alt_unit": "Mtrs", "alt_qty": "-5", "qty": "15.00"},
            ],
        }

        def mock_match(raw, master_key, client_id, parent_id=None):
            return MatchResult(raw=raw, matched_id=10, matched_label="Mtrs", confidence=1.0, master=master_key)

        with patch("app.main.indexer.is_loaded", return_value=True), \
             patch("app.main.match_entity_exact", side_effect=mock_match):

            res = client.post(
                "/api/v1/invoice/process",
                headers={"X-API-Key": API_KEY},
                data={
                    "client_id": "1",
                    "output_schema": schema,
                    "extracted_edited_json_by_user": json.dumps(extracted_data),
                },
            )

        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["line_items"]) == 2
        assert body["warning_message"] is not None
        assert "Row 1: Alt Qty ('0') must be greater than 0" in body["warning_message"]
        assert "Row 2: Alt Qty ('-5') must be greater than 0" in body["warning_message"]

