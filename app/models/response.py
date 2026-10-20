"""
app/models/response.py
──────────────────────
Pydantic models for the API response.  All field result types are strict
unions so the serialised JSON always matches the documented API contract.

New in v2 (stepped workflow):
  OCRExtractResponse  — returned by POST /api/v1/invoice/extract
  ProcessingError     — 422 payload when auto_search fields cannot be resolved
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Per-field result models ───────────────────────────────────────────────────


class DirectFieldResult(BaseModel):
    """
    Result for a field with type == "direct".

    Attributes:
        type: Always "direct".
        value: Extracted value (string, numeric, boolean, etc.), or None if not found in document.
        confidence: Model confidence in [0.0, 1.0].
    """

    type: Literal["direct"] = "direct"
    value: Optional[Any] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AlternativeMatch(BaseModel):
    """
    A runner-up candidate from fuzzy entity matching.

    Attributes:
        id: Primary key in the master table.
        label: Human-readable entity name.
        confidence: Normalised fuzzy score in [0.0, 1.0].
    """

    id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class AutoSearchFieldResult(BaseModel):
    """
    Result for a field with type == "auto_search".

    Attributes:
        type: Always "auto_search".
        master: The entity index key used for matching.
        raw: Raw text extracted from the OCR output for this field,
             or None if the field was not present in the document.
        matched_id: Primary key of the best match, or None if confidence < 0.60.
        matched_label: Human-readable label of the best match.
        confidence: Normalised fuzzy score of the best match.
        alternatives: Up to 2 runner-up matches above 60 % confidence.
        parent_field: Populated when the field had a parent dependency.
    """

    type: Literal["auto_search"] = "auto_search"
    master: str
    raw: Optional[str] = None
    matched_id: Optional[int] = None
    matched_label: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: List[AlternativeMatch] = Field(default_factory=list)
    parent_field: Optional[str] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class AutoPopulatedFieldResult(BaseModel):
    """
    Result for a field with type == "auto_populated".

    These fields are populated server-side by the ERP after entity matching;
    the microservice skips them entirely and always returns value = None.

    Attributes:
        type: Always "auto_populated".
        value: Always None.
    """

    type: Literal["auto_populated"] = "auto_populated"
    value: None = None


class LineItemsFieldResult(BaseModel):
    """
    Result for a field with type == "line_items".

    Contains a list of dicts, one per row extracted from the document's
    dispatch / item table.  Each dict uses the keys declared in item_fields.

    Attributes:
        type: Always "line_items".
        items: List of extracted row dicts.  Empty list if no rows found.
        confidence: Overall extraction confidence across all rows.
    """

    type: Literal["line_items"] = "line_items"
    items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# Union type used in the response data dict
FieldResult = Union[
    DirectFieldResult,
    AutoSearchFieldResult,
    AutoPopulatedFieldResult,
    LineItemsFieldResult,
]


# ── Response meta ─────────────────────────────────────────────────────────────


class ResponseMeta(BaseModel):
    """
    Metadata attached to every successful response.

    Attributes:
        model: OCR model used.
        pages: Number of pages processed.
        scan_cost_usd: Estimated cost of the OCR call in USD.
        processed_at: UTC timestamp of when processing completed.
    """

    model: str
    pages: int
    scan_cost_usd: float
    processed_at: datetime


# ── Top-level response ────────────────────────────────────────────────────────


class InvoiceParseResponse(BaseModel):
    """
    Full response envelope for POST /api/v1/invoice/parse.

    Attributes:
        status: Always "success" on HTTP 200.
        data: Dict mapping each schema field name to its extraction result.
        meta: Processing metadata.
        warning_message: Optional human-readable warning message when auto_search exact match fails.
    """

    status: Literal["success"] = "success"
    data: Dict[str, FieldResult]
    meta: ResponseMeta
    warning_message: Optional[str] = None


class SectionedInvoiceParseResponse(BaseModel):
    """
    Nested response envelope for POST /api/v1/invoice/parse.

    Returns extracted values grouped into three sections, with plain scalar
    values (no confidence wrappers) ready for direct ERP consumption.

    Attributes:
        status: Always "success" on HTTP 200.
        basic_details: Header-level fields extracted from the document.
                       auto_search fields resolve to the matched entity label.
                       direct fields return the raw extracted string/number.
        line_items: List of row dicts extracted from the dispatch/item table.
                    One dict per physical row, keys defined by item_fields.
        summary: Aggregate / footer fields extracted from the document.
        meta: Processing metadata (model, pages, cost, timestamp).
        warning_message: Optional human-readable warning message when auto_search exact match fails.
    """

    status: Literal["success"] = "success"
    basic_details: Dict[str, Any] = Field(default_factory=dict)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    meta: ResponseMeta
    warning_message: Optional[str] = None


# ── OCR extract response ─────────────────────────────────────────────────────


class OCRExtractResponse(BaseModel):
    """
    Response for POST /api/v1/invoice/extract (Step 1 of the stepped workflow).

    The caller receives the raw OCR text and can present it to the user for
    review / correction before submitting it to /api/v1/invoice/process.

    Attributes:
        status: Always "extracted" on HTTP 200.
        raw_text: Full text extracted by Mistral OCR (all files concatenated).
        page_count: Total number of pages processed across all files.
        file_name: Comma-joined display string of uploaded file names.
        file_names: List of individual uploaded file names.
        scan_cost_usd: Estimated total OCR cost in USD.
    """

    status: Literal["extracted"] = "extracted"
    raw_text: str
    extracted_json: Optional[Dict[str, Any]] = None
    page_count: int
    file_name: str                           # backward-compat: comma-joined
    file_names: List[str] = Field(default_factory=list)   # full list
    scan_cost_usd: float



# ── Error response ────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """
    Structured error payload returned on non-200 responses.

    Attributes:
        error: Short error code string (snake_case).
        message: Human-readable description of the problem.
        request_id: UUID from the X-Request-ID header, for log correlation.
    """

    error: str
    message: str
    request_id: Optional[str] = None


class ProcessingError(BaseModel):
    """
    Error payload returned when one or more auto_search fields cannot be
    resolved to an exact match in the entity index.

    Attributes:
        error: Always "entity_not_found".
        message: Human-readable summary listing all unresolved fields.
        unresolved_fields: List of field names that had no exact match.
        request_id: UUID from X-Request-ID header.
    """

    error: Literal["entity_not_found"] = "entity_not_found"
    message: str
    unresolved_fields: List[str]
    request_id: Optional[str] = None


# ── Health check response ─────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """
    Response for GET /health.

    Attributes:
        status: "ok" when the service is healthy.
        index_loaded: True once the entity index has been populated.
        indexed_clients: Number of distinct client_ids in the entity index.
        uptime_seconds: Seconds since the FastAPI application started.
    """

    status: Literal["ok"] = "ok"
    index_loaded: bool
    indexed_clients: int
    uptime_seconds: float


# ── Index refresh response ────────────────────────────────────────────────────


class IndexRefreshResponse(BaseModel):
    """
    Response for GET /api/v1/index/refresh.

    Attributes:
        status: "refreshed" on success.
        indexed_clients: Number of client records now in the index.
        message: Human-readable confirmation.
    """

    status: Literal["refreshed"] = "refreshed"
    indexed_clients: int
    message: str
