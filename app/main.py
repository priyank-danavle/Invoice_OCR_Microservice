"""
app/main.py
───────────
FastAPI application entry point.

Registers:
  - CORS middleware (origins from .env)
  - X-Request-ID middleware (UUID on every response)
  - Global exception handlers (never expose raw tracebacks)
  - Structured JSON request logging
  - Startup/shutdown lifecycle hooks (DB pool + entity index)

Routes:
  POST /api/v1/invoice/parse    → main parsing endpoint
  GET  /health                  → liveness / readiness probe
  GET  /api/v1/index/refresh    → manual entity index reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import settings
from app.db import indexer
from app.models.enums import InwardType
from app.models.request import OutputSchema, SchemaField, parse_output_schema
from app.ocr.inward_classifier import determine_inward_type_from_dict, normalize_inward_type_value
from app.models.response import (
    AlternativeMatch,
    AutoPopulatedFieldResult,
    AutoSearchFieldResult,
    DirectFieldResult,
    ErrorDetail,
    HealthResponse,
    IndexRefreshResponse,
    InvoiceParseResponse,
    LineItemsFieldResult,
    OCRExtractResponse,
    ProcessingError,
    ResponseMeta,
    SectionedInvoiceParseResponse,
)
from app.ocr.challan_validator import MultipleChallansDetectedError, check_and_raise_single_challan
from app.ocr.extractor import OCRExtractionError, extract
from app.ocr.matcher import EntityMatchError, MatchResult, match_entity, match_entity_exact
from app.ocr.parser import ParserError, parse
from app.utils.file_handler import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    file_size_kb,
    read_and_validate,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application startup time ──────────────────────────────────────────────────

_startup_time: float = time.monotonic()

# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="OCR Invoice Reader",
    description=(
        "AI-powered document parsing microservice for the textile ERP. "
        "Extracts structured fields from invoices and purchase orders using "
        "Mistral OCR + fuzzy entity matching against ERP master data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

# ── Static UI ─────────────────────────────────────────────────────────────────
import os as _os
_ui_dir = _os.path.join(_os.path.dirname(__file__), "..", "ui")
if _os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root to the OCR UI dashboard."""
    return RedirectResponse(url="/ui/index.html")

# ── CORS middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── X-Request-ID middleware ───────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach a UUID X-Request-ID header to every request and response.
    The request_id is stored in request.state for use in log lines.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class PathNormalizationMiddleware(BaseHTTPMiddleware):
    """
    Normalizes request paths to eliminate 404 errors caused by:
    - Trailing slashes (e.g., /api/v1/invoice/parse/ -> /api/v1/invoice/parse)
    - Subpath prefixes passed from upstream reverse proxies (e.g., /ocr-api/api/v1/invoice/parse -> /api/v1/invoice/parse)
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.scope.get("path", "")
        if path.startswith("/ocr-api/") or path == "/ocr-api":
            path = path[len("/ocr-api"):]
            if not path:
                path = "/"
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        request.scope["path"] = path
        return await call_next(request)


app.add_middleware(PathNormalizationMiddleware)
app.add_middleware(RequestIDMiddleware)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler — never expose raw tracebacks to the client."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception | request_id=%s path=%s: %s",
        request_id, request.url.path, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error="internal_server_error",
            message="An unexpected error occurred. Please try again or contact support.",
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured error JSON for all HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "HTTPException %d | request_id=%s path=%s: %s",
        exc.status_code, request_id, request.url.path, exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorDetail(
            error=_status_to_error_code(exc.status_code),
            message=exc.detail,
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(MultipleChallansDetectedError)
async def multiple_challans_detected_handler(request: Request, exc: MultipleChallansDetectedError) -> JSONResponse:
    """Return structured HTTP 400 JSON when multiple delivery challans are detected."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "MultipleChallansDetectedError | request_id=%s path=%s: %s",
        request_id, request.url.path, exc.message,
    )
    return JSONResponse(
        status_code=400,
        content=ErrorDetail(
            error="multiple_challans_detected",
            message=exc.message,
            request_id=request_id,
        ).model_dump(),
    )


def _status_to_error_code(status: int) -> str:
    codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "unprocessable_content",
        429: "rate_limited",
        500: "internal_server_error",
    }
    return codes.get(status, "error")


# ── Lifecycle hooks ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Load entity index and start background refresh on application startup."""
    global _startup_time
    _startup_time = time.monotonic()
    logger.info("OCR Invoice Reader starting up …")
    try:
        await indexer.load()
        indexer.start_background_refresh()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Entity index failed to load on startup: %s. "
            "Service will start but entity matching will be unavailable.",
            exc,
        )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cancel background tasks and release DB connections on shutdown."""
    logger.info("OCR Invoice Reader shutting down …")
    indexer.stop_background_refresh()
    await indexer.close_pool()


# ── Auth dependency ───────────────────────────────────────────────────────────

def verify_api_key(request: Request) -> None:
    """
    FastAPI dependency that validates the X-API-Key header.

    Raises:
        HTTPException: HTTP 401 if header is missing or does not match
                       MICROSERVICE_API_KEY from .env.
    """
    logger.info("headers: %s", request.headers)
    api_key = request.headers.get("x-api-key", "")
    if not api_key or api_key != settings.MICROSERVICE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header.",
        )


# ── Topological sort for field dependencies ───────────────────────────────────

def _topological_sort_fields(schema: OutputSchema) -> list[str]:
    """
    Return field names in dependency order (parents before children).

    Non-auto_search fields have no dependencies and come first.
    auto_search fields with a parent_field are placed after their parent.

    Args:
        schema: Validated output schema from the ERP backend.

    Returns:
        list[str]: Field names in safe processing order.
    """
    resolved: list[str] = []
    remaining = list(schema.keys())

    # Separate: fields with no parent dependency
    no_dep = [f for f in remaining if not schema[f].parent_field]
    with_dep = [f for f in remaining if schema[f].parent_field]

    resolved.extend(no_dep)

    # Simple single-pass for the dependency layer (only one level deep in spec)
    for field_name in with_dep:
        parent = schema[field_name].parent_field
        if parent and parent not in resolved:
            # Parent not yet in resolved — insert parent first (if it exists)
            if parent in schema and parent not in resolved:
                resolved.append(parent)
                if parent in with_dep:
                    with_dep.remove(parent)
        resolved.append(field_name)

    return resolved


# ── Main parse route ──────────────────────────────────────────────────────────

@app.post(
    "/api/v1/invoice/parse",
    response_model=SectionedInvoiceParseResponse,
    summary="Parse invoice document",
    description=(
        "Upload a JPG/PNG/PDF invoice, supply a client_id and output_schema, "
        "and receive structured field extraction results grouped into "
        "basic_details, line_items, and summary sections."
    ),
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/api/v1/invoice/parse/",
    response_model=SectionedInvoiceParseResponse,
    include_in_schema=False,
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/ocr-api/api/v1/invoice/parse",
    response_model=SectionedInvoiceParseResponse,
    include_in_schema=False,
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/ocr-api/api/v1/invoice/parse/",
    response_model=SectionedInvoiceParseResponse,
    include_in_schema=False,
    dependencies=[Depends(verify_api_key)],
)
async def parse_invoice(
    request: Request,
    file: List[UploadFile] | UploadFile = File(
        ...,
        description="Invoice image(s) or PDF(s) (max 10 MB each).",
    ),
    client_id: int = Form(..., description="ERP client identifier."),
    output_schema: str = Form(..., description="JSON string defining extraction schema."),
) -> SectionedInvoiceParseResponse:
    """
    Main document parsing endpoint.

    Steps:
        1. Validate API key (handled by dependency).
        2. Validate file type and size.
        3. Validate client_id presence (already enforced by Form type int).
        4. Parse and validate output_schema JSON.
        5. Run Mistral OCR → raw_text.
        6. Run Mistral field parser → extracted values.
        7. Topologically sort fields, then run fuzzy matching for auto_search.
        8. Assemble response.

    Returns:
        InvoiceParseResponse: Structured extraction results.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")
    start_time = time.monotonic()

    try:
        # ── 1. Validate file(s) ───────────────────────────────────────────────
        form = await request.form()
        files_input: List[UploadFile] = []
        for _, val in form.multi_items():
            if isinstance(val, UploadFile) and val.filename:
                files_input.append(val)
        if not files_input and file:
            files_input = file if isinstance(file, list) else [file]
        if not files_input:
            raise HTTPException(status_code=400, detail="At least one file is required.")

        validated: List[tuple] = []
        for f in files_input:
            try:
                fb, mt = await read_and_validate(f)
                validated.append((f, fb, mt))
            except UnsupportedFileTypeError as exc:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}': {exc}")
            except FileTooLargeError as exc:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}': {exc}")
            except EmptyFileError as exc:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}': {exc}")

        # ── 2. Validate client_id ─────────────────────────────────────────────
        if client_id <= 0:
            raise HTTPException(
                status_code=400,
                detail="client_id must be a positive integer.",
            )

        # ── 3. Parse output_schema ────────────────────────────────────────────
        try:
            parsed_schema: OutputSchema = parse_output_schema(output_schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"output_schema is not valid JSON: {exc.msg}",
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"output_schema validation error: {exc}",
            )

        if not parsed_schema:
            raise HTTPException(
                status_code=400,
                detail="output_schema must contain at least one field definition.",
            )

        total_kb = sum(file_size_kb(fb) for _, fb, _ in validated)
        logger.info(
            "Parse request: request_id=%s client_id=%d files=%d size_kb=%.1f fields=%d",
            request_id, client_id, len(validated), total_kb, len(parsed_schema),
        )

        # ── 4. OCR extraction ─────────────────────────────────────────────────
        try:
            raw_text, page_count = await extract([(fb, mt) for _, fb, mt in validated])
        except OCRExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # ── 5. Field parsing ──────────────────────────────────────────────────
        try:
            parsed_values: Dict[str, Dict[str, Any]] = await parse(raw_text, parsed_schema, client_id=client_id)
        except ParserError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
            
        import json
        with open("debug_parsed_data.json", "w") as f:
            json.dump(parsed_values, f, indent=4)
            
        logger.info(
            "Parsed %d fields from OCR output %s | request_id=%s",
            len(parsed_values), parsed_values, request_id,
        )
        # ── 6. Build result map ───────────────────────────────────────────────
        field_order = _topological_sort_fields(parsed_schema)

        # Track matched IDs for parent → child dependency resolution
        matched_ids: Dict[str, int | None] = {}
        result_data: Dict[str, Any] = {}

        for field_name in field_order:
            schema_field: SchemaField = parsed_schema[field_name]

            # ── inward_type → calculated classification result ─────────────────────
            if field_name == "inward_type":
                inward_raw = None
                field_extracted = parsed_values.get("inward_type", {})
                if isinstance(field_extracted, dict):
                    inward_raw = field_extracted.get("value")
                elif field_extracted is not None:
                    inward_raw = field_extracted

                inward_val = normalize_inward_type_value(inward_raw)
                if inward_val is None:
                    inward_val = determine_inward_type_from_dict(parsed_values, client_id=client_id).value

                result_data["inward_type"] = DirectFieldResult(
                    value=inward_val,
                    confidence=1.0,
                )
                continue

            # ── auto_populated → always null ──────────────────────────────────
            if schema_field.type == "auto_populated":
                result_data[field_name] = AutoPopulatedFieldResult()
                continue

            field_extracted = parsed_values.get(field_name, {})
            value = field_extracted.get("value")
            confidence = field_extracted.get("confidence", 0.0)

            # ── line_items → list of dicts ────────────────────────────────────
            if schema_field.type == "line_items":
                items = value if isinstance(value, list) else []
                result_data[field_name] = LineItemsFieldResult(
                    items=items,
                    confidence=confidence if items else 0.0,
                )
                continue

            # ── direct → return extracted value ───────────────────────────────
            if schema_field.type == "direct":
                result_data[field_name] = DirectFieldResult(
                    value=value,
                    confidence=confidence if value is not None else 0.0,
                )
                continue

            # ── auto_search → fuzzy match ─────────────────────────────────────
            parent_id: int | None = None
            if schema_field.parent_field and schema_field.parent_field in matched_ids:
                parent_id = matched_ids[schema_field.parent_field]

            try:
                match: MatchResult = match_entity(
                    raw=value,
                    master_key=schema_field.master,  # type: ignore[arg-type]
                    client_id=client_id,
                    parent_id=parent_id,
                )
            except EntityMatchError as exc:
                if "client_id" in str(exc) and "not found" in str(exc):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid client_id {client_id}: client not found in entity index.",
                    )
                logger.warning(
                    "Entity match fallback for field '%s' (master='%s', client_id=%d) | request_id=%s: %s",
                    field_name, schema_field.master, client_id, request_id, exc,
                )
                match = MatchResult(
                    raw=value,
                    matched_id=None,
                    matched_label=None,
                    confidence=0.0,
                    master=schema_field.master or "",
                )

            # Record matched_id for downstream child fields
            matched_ids[field_name] = match.matched_id

            result_data[field_name] = AutoSearchFieldResult(
                master=schema_field.master or "",
                raw=match.raw,
                matched_id=match.matched_id,
                matched_label=match.matched_label,
                confidence=match.confidence,
                alternatives=[
                    AlternativeMatch(
                        id=alt.id,
                        label=alt.label,
                        confidence=alt.confidence,
                    )
                    for alt in match.alternatives
                ],
                parent_field=schema_field.parent_field,
                extra_data=match.extra_data,
            )

        # ── 7. Assemble sectioned response ────────────────────────────────────
        processing_ms = int((time.monotonic() - start_time) * 1000)
        scan_cost = round(page_count * settings.COST_PER_PAGE_USD, 6)

        basic_details: Dict[str, Any] = {}
        line_items_out: List[Dict[str, Any]] = []
        summary_out: Dict[str, Any] = {}

        for field_name, field_result in result_data.items():
            schema_field = parsed_schema[field_name]

            # line_items type → resolve auto_search columns, then list section
            if isinstance(field_result, LineItemsFieldResult):
                raw_rows = field_result.items
                li_auto_search = parsed_schema[field_name].get_line_item_auto_search_fields()

                resolved_rows: List[Dict[str, Any]] = []
                for row in raw_rows:
                    resolved_row = dict(row)

                    # Custom textile logic: split description_of_goods if it follows the composite format
                    # Format: "quality : design : chart : color"
                    # The description is always the authoritative source for all 4 fields.
                    # The LLM often extracts wrong values for design_master (e.g. a row number
                    # like "1" instead of "glossy") so we always overwrite from the description
                    # split — consistent with how color_master is already handled.
                    # Fill missing/empty master fields from description_of_goods split only if not already provided
                    desc = resolved_row.get("description_of_goods")
                    if isinstance(desc, str) and ":" in desc:
                        parts = [p.strip() for p in desc.split(":")]
                        if len(parts) >= 1 and not resolved_row.get("quality_master"):
                            resolved_row["quality_master"] = parts[0]
                        if len(parts) >= 2 and not resolved_row.get("design_master"):
                            resolved_row["design_master"] = parts[1]
                        if len(parts) >= 3 and not resolved_row.get("chart_master"):
                            resolved_row["chart_master"] = parts[2]
                        if len(parts) >= 4 and not resolved_row.get("color_master"):
                            resolved_row["color_master"] = parts[3]

                    for col, master_key in li_auto_search.items():
                        raw_val = resolved_row.get(col)
                        if isinstance(raw_val, dict):
                            raw_val = raw_val.get("label") or raw_val.get("value") or raw_val.get("name") or (str(raw_val.get("id")) if raw_val.get("id") is not None else None)
                        try:
                            m: MatchResult = match_entity(
                                raw=raw_val,
                                master_key=master_key,
                                client_id=client_id,
                                parent_id=None,
                            )
                            resolved_row[col] = {
                                "id": m.matched_id,
                                "label": m.matched_label,
                                **m.extra_data,
                            }
                        except EntityMatchError:
                            resolved_row[col] = {"id": None, "label": raw_val}  # fall back to raw text
                    # Normalize line item quantity if extracted as thousands integer due to stripped decimal (e.g. 20000 -> 20.0)
                    for qty_col in ("supplier_qty", "qty", "quantity"):
                        if qty_col in resolved_row and resolved_row[qty_col] is not None:
                            val = resolved_row[qty_col]
                            if isinstance(val, (int, float)) and val >= 1000 and val % 1000 == 0:
                                resolved_row[qty_col] = round(val / 1000.0, 3)
                            elif isinstance(val, str) and val.isdigit() and int(val) >= 1000 and int(val) % 1000 == 0:
                                resolved_row[qty_col] = round(int(val) / 1000.0, 3)

                    resolved_rows.append(resolved_row)

                line_items_out = resolved_rows
                continue

            # Extract plain scalar value from each result type
            if isinstance(field_result, DirectFieldResult):
                plain_value: Any = field_result.value
            elif isinstance(field_result, AutoSearchFieldResult):
                if field_result.matched_id is not None:
                    plain_value = {
                        "id": field_result.matched_id,
                        "label": field_result.matched_label,
                        **field_result.extra_data,
                    }
                else:
                    raw_val = field_result.raw
                    if isinstance(raw_val, dict):
                        raw_val = raw_val.get("label") or raw_val.get("value") or raw_val.get("name") or (str(raw_val.get("id")) if raw_val.get("id") is not None else None)
                    plain_value = raw_val if raw_val is not None else ""
            elif isinstance(field_result, AutoPopulatedFieldResult):
                plain_value = None
            else:
                plain_value = None

            # Route to the correct section
            if schema_field.section == "summary":
                summary_out[field_name] = plain_value
            else:
                basic_details[field_name] = plain_value

        file_names_str = ", ".join(f.filename or "unknown" for f, _, _ in validated)
        logger.info(
            json.dumps({
                "request_id": request_id,
                "client_id": client_id,
                "file_name": file_names_str,
                "file_size_kb": total_kb,
                "pages": page_count,
                "processing_time_ms": processing_ms,
                "scan_cost_usd": scan_cost,
                "status": "success",
            })
        )

        return SectionedInvoiceParseResponse(
            status="success",
            basic_details=basic_details,
            line_items=line_items_out,
            summary=summary_out,
            meta=ResponseMeta(
                model=settings.MISTRAL_OCR_MODEL,
                pages=page_count,
                scan_cost_usd=scan_cost,
                processed_at=datetime.now(tz=timezone.utc),
            ),
        )

    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions (they have handlers above)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unhandled error in parse_invoice | request_id=%s: %s",
            request_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during invoice processing.",
        ) from exc


# ── Step 1: OCR Extract ───────────────────────────────────────────────────────

@app.post(
    "/api/v1/invoice/extract",
    response_model=OCRExtractResponse,
    summary="[Step 1] Extract raw OCR text from document(s)",
    description=(
        "Upload one or more JPG/PNG/PDF invoices and receive the raw text extracted by "
        "Mistral OCR. Multiple files are processed in parallel and their text is "
        "concatenated with clear file separators. Present this text to the user for "
        "review / correction, then pass the (optionally edited) text to "
        "POST /api/v1/invoice/process (Step 2)."
    ),
    dependencies=[Depends(verify_api_key)],
)
@app.post("/api/v1/invoice/extract/", response_model=OCRExtractResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
@app.post("/ocr-api/api/v1/invoice/extract", response_model=OCRExtractResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
@app.post("/ocr-api/api/v1/invoice/extract/", response_model=OCRExtractResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
async def extract_invoice(
    request: Request,
    file: List[UploadFile] = File(
        ...,
        description="Invoice image(s) or PDF(s) (max 10 MB each). "
                    "Send one or more 'file' fields to process several documents at once.",
    ),
    client_id: int = Form(..., description="ERP client identifier."),
    output_schema: Optional[str] = Form(
        None,
        description="Optional JSON schema string. If provided, initial field extraction is "
                    "executed and extracted fields are returned as formatted JSON in raw_text & extracted_json.",
    ),
) -> OCRExtractResponse:
    """
    Step 1 of the stepped OCR workflow.

    Accepts one or more invoice files. All files are validated, then sent to
    Mistral OCR in parallel. If an output_schema is provided, initial field
    extraction is executed and the extracted fields are formatted as JSON in
    raw_text (and returned in extracted_json) so the frontend can preview and edit JSON directly.

    Returns:
        OCRExtractResponse: Combined raw text (or formatted JSON string), total page count,
                            file names, extracted_json object, and total scan cost.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")

    if client_id <= 0:
        raise HTTPException(status_code=400, detail="client_id must be a positive integer.")

    # ── Validate every uploaded file eagerly ──────────────────────────────────
    form = await request.form()
    files_input: List[UploadFile] = []
    for _, val in form.multi_items():
        if isinstance(val, UploadFile) and val.filename:
            files_input.append(val)
    if not files_input and file:
        files_input = file if isinstance(file, list) else [file]
    if not files_input:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    validated: List[tuple] = []   # (UploadFile, bytes, mime_type)
    for f in files_input:
        try:
            file_bytes, mime_type = await read_and_validate(f)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}': {exc}",
            )
        except FileTooLargeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}': {exc}",
            )
        except EmptyFileError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}': {exc}",
            )
        validated.append((f, file_bytes, mime_type))

    logger.info(
        "Extract request: request_id=%s client_id=%d files=%d names=%s has_schema=%s",
        request_id, client_id, len(validated),
        [f.filename for f, _, _ in validated],
        bool(output_schema and output_schema.strip()),
    )

    # ── Run OCR on all uploaded files ─────────────────────────────────────────
    try:
        combined_text, total_pages = await extract([(fb, mt) for _, fb, mt in validated])
    except OCRExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"OCR failed: {exc}",
        )

    file_names = [f.filename or "unknown" for f, _, _ in validated]
    scan_cost = round(total_pages * settings.COST_PER_PAGE_USD, 6)

    # Validate that only a single delivery challan was uploaded
    check_and_raise_single_challan(combined_text)

    extracted_json_dict: Optional[Dict[str, Any]] = None
    final_raw_text = combined_text

    # If output_schema is provided, run initial LLM parsing to populate JSON preview
    if output_schema and output_schema.strip():
        try:
            parsed_schema: OutputSchema = parse_output_schema(output_schema)
            if parsed_schema:
                pv = await parse(combined_text, parsed_schema, client_id=client_id)
                clean_dict: Dict[str, Any] = {}
                for k, v in pv.items():
                    if isinstance(v, dict) and "value" in v:
                        clean_dict[k] = v["value"]
                    else:
                        clean_dict[k] = v
                extracted_json_dict = clean_dict
                final_raw_text = json.dumps(clean_dict, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(
                "Extract: field parsing with output_schema failed, falling back to raw OCR text | request_id=%s: %s",
                request_id, exc,
            )

    logger.info(
        "Extract complete: request_id=%s files=%d pages=%d cost=$%.6f parse_json=%s",
        request_id, len(validated), total_pages, scan_cost, extracted_json_dict is not None,
    )

    return OCRExtractResponse(
        raw_text=final_raw_text,
        extracted_json=extracted_json_dict,
        page_count=total_pages,
        file_name=", ".join(file_names),
        file_names=file_names,
        scan_cost_usd=scan_cost,
    )


def _is_zero_or_empty_qty(val: Any) -> bool:
    """Check if quantity value is None, empty, zero, negative, or dash."""
    if val is None:
        return True
    if isinstance(val, (int, float)) and val <= 0:
        return True
    val_str = str(val).strip()
    if not val_str or val_str in ("-", "null", "None"):
        return True
    try:
        if float(val_str) <= 0:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _is_alt_unit_col(col_name: str) -> bool:
    """Check if line item column represents an alternative unit."""
    c = col_name.lower()
    return "alt_unit" in c or "altunit" in c or "alt_uom" in c or c in ("secondary_unit", "alternative_unit")


def _is_alt_qty_col(col_name: str) -> bool:
    """Check if line item column represents an alternative quantity."""
    c = col_name.lower()
    return "alt_qty" in c or "altqty" in c or "alt_quantity" in c or c in ("secondary_qty", "alternative_qty")


def _find_alt_qty_col(alt_unit_col: str, row: Dict[str, Any], item_fields: Dict[str, Any]) -> str | None:
    """Find corresponding alt quantity column name for an alt unit column."""
    unit_c = alt_unit_col.lower()
    qty_target = unit_c.replace("alt_unit", "alt_qty").replace("altunit", "altqty").replace("alt_uom", "alt_qty").replace("unit", "qty")
    all_keys = list(dict.fromkeys(list(row.keys()) + list(item_fields.keys())))
    for k in all_keys:
        if k.lower() == qty_target:
            return k
    for k in all_keys:
        if _is_alt_qty_col(k):
            return k
    return None


# ── Step 2: Process (field parse + exact match) ───────────────────────────────

@app.post(
    "/api/v1/invoice/process",
    response_model=SectionedInvoiceParseResponse,
    summary="[Step 2] Parse fields and exact-match entities from OCR text",
    description=(
        "Takes the raw OCR text or JSON string (optionally user-corrected from Step 1), "
        "parses/extracts structured fields, then performs EXACT entity matching "
        "against ERP master data. "
        "If ANY auto_search field cannot be resolved to an exact match the "
        "request fails with HTTP 422 listing all unresolved field names."
    ),
    dependencies=[Depends(verify_api_key)],
)
@app.post("/api/v1/invoice/process/", response_model=SectionedInvoiceParseResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
@app.post("/ocr-api/api/v1/invoice/process", response_model=SectionedInvoiceParseResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
@app.post("/ocr-api/api/v1/invoice/process/", response_model=SectionedInvoiceParseResponse, include_in_schema=False, dependencies=[Depends(verify_api_key)])
async def process_invoice(
    request: Request,
    client_id: Optional[Any] = Form(None, description="ERP client identifier."),
    output_schema: Optional[Any] = Form(None, description="JSON string or dict defining extraction schema."),
    outschema: Optional[Any] = Form(None, description="Alias for output_schema."),
    raw_text: Optional[Any] = Form(None, description="OCR text or JSON string."),
    extracted_edited_json_by_user: Optional[Any] = Form(None, description="User-edited JSON string or dict from Step 1."),
) -> SectionedInvoiceParseResponse:
    """
    Step 2 of the stepped OCR workflow.

    Accepts raw OCR text or user-edited JSON (via application/json or multipart/form-data),
    parses structured fields from it, then performs strict exact-match
    entity lookup for all auto_search fields.

    If any auto_search field (top-level or inside line_items) fails to resolve
    to an exact entity the entire request is rejected with HTTP 422 and a
    payload listing every unresolved field so the user knows what to correct.

    Returns:
        SectionedInvoiceParseResponse: Same shape as /api/v1/invoice/parse.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")
    start_time = time.monotonic()

    # Initial values from Form inputs
    req_client_id = client_id
    req_schema = output_schema if output_schema is not None else outschema
    req_extracted = extracted_edited_json_by_user if extracted_edited_json_by_user is not None else raw_text

    # Check for application/json content-type or attempt reading JSON body
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type.lower() or not (req_client_id and req_schema and req_extracted):
        try:
            body = await request.json()
            if isinstance(body, dict):
                if "client_id" in body:
                    req_client_id = body["client_id"]
                if "outschema" in body:
                    req_schema = body["outschema"]
                elif "output_schema" in body:
                    req_schema = body["output_schema"]
                if "extracted_edited_json_by_user" in body:
                    req_extracted = body["extracted_edited_json_by_user"]
                elif "raw_text" in body:
                    req_extracted = body["raw_text"]
        except Exception as exc:
            logger.debug("Attempt to parse JSON request body yielded: %s", exc)

    # ── Validate & Parse client_id ───────────────────────────────────────────
    if req_client_id is None:
        raise HTTPException(status_code=400, detail="client_id is required.")

    try:
        final_client_id = int(req_client_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="client_id must be a valid integer.")

    if final_client_id <= 0:
        raise HTTPException(status_code=400, detail="client_id must be a positive integer.")

    # Override for downstream entity matching logic
    client_id = final_client_id

    # ── Validate & Parse output_schema ───────────────────────────────────────
    if req_schema is None:
        raise HTTPException(status_code=400, detail="output_schema (or outschema) is required.")

    if isinstance(req_schema, dict):
        schema_str = json.dumps(req_schema)
    elif isinstance(req_schema, str):
        schema_str = req_schema
    else:
        raise HTTPException(status_code=400, detail="output_schema must be a JSON object or JSON string.")

    try:
        parsed_schema: OutputSchema = parse_output_schema(schema_str)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"output_schema is not valid JSON: {exc.msg}",
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"output_schema validation error: {exc}",
        )

    if not parsed_schema:
        raise HTTPException(
            status_code=400,
            detail="output_schema must contain at least one field definition.",
        )

    # ── Validate & Parse extracted data / raw text ───────────────────────────
    if req_extracted is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'extracted_edited_json_by_user' or 'raw_text' is required.",
        )

    logger.info(
        "Process request: request_id=%s client_id=%d fields=%d extracted_type=%s",
        request_id, client_id, len(parsed_schema), type(req_extracted).__name__,
    )

    parsed_values: Dict[str, Dict[str, Any]] | None = None

    if isinstance(req_extracted, dict):
        parsed_values = {}
        for k, v in req_extracted.items():
            if isinstance(v, dict) and "value" in v and "confidence" in v:
                parsed_values[k] = v
            else:
                parsed_values[k] = {"value": v, "confidence": 1.0}
        logger.info(
            "Process: using direct JSON object input (%d fields) | request_id=%s",
            len(parsed_values), request_id,
        )
    elif isinstance(req_extracted, str):
        cleaned_input = req_extracted.strip()
        if not cleaned_input:
            raise HTTPException(status_code=400, detail="extracted data / raw_text cannot be empty.")
        if cleaned_input.startswith("{") or cleaned_input.startswith("["):
            try:
                data = json.loads(cleaned_input)
                if isinstance(data, dict):
                    parsed_values = {}
                    for k, v in data.items():
                        if isinstance(v, dict) and "value" in v and "confidence" in v:
                            parsed_values[k] = v
                        else:
                            parsed_values[k] = {"value": v, "confidence": 1.0}
                    logger.info(
                        "Process: using direct JSON string input (%d fields) | request_id=%s",
                        len(parsed_values), request_id,
                    )
            except Exception as exc:
                logger.debug("Process: input string is JSON-like but not valid dict: %s", exc)

        if parsed_values is None:
            try:
                parsed_values = await parse(cleaned_input, parsed_schema, client_id=client_id)
                logger.info(
                    "Process: parsed %d fields via LLM | request_id=%s",
                    len(parsed_values), request_id,
                )
            except ParserError as exc:
                raise HTTPException(status_code=500, detail=str(exc))
    else:
        raise HTTPException(status_code=400, detail="extracted_edited_json_by_user / raw_text must be a string or JSON object.")

    logger.info(
        "Process: parsed %d fields | request_id=%s",
        len(parsed_values), request_id,
    )

    # ── Exact entity matching ─────────────────────────────────────────────────
    field_order = _topological_sort_fields(parsed_schema)
    matched_ids: Dict[str, int | None] = {}
    result_data: Dict[str, Any] = {}
    warnings_list: List[str] = []

    for field_name in field_order:
        schema_field: SchemaField = parsed_schema[field_name]

        # ── inward_type → calculated classification result ─────────────────────
        if field_name == "inward_type":
            inward_raw = None
            field_extracted = parsed_values.get("inward_type", {})
            if isinstance(field_extracted, dict):
                inward_raw = field_extracted.get("value")
            elif field_extracted is not None:
                inward_raw = field_extracted

            inward_val = normalize_inward_type_value(inward_raw)
            if inward_val is None:
                inward_val = determine_inward_type_from_dict(parsed_values, client_id=client_id).value

            result_data["inward_type"] = DirectFieldResult(
                value=inward_val,
                confidence=1.0,
            )
            continue

        # auto_populated → always null
        if schema_field.type == "auto_populated":
            result_data[field_name] = AutoPopulatedFieldResult()
            continue

        field_extracted = parsed_values.get(field_name, {})
        value = field_extracted.get("value")
        confidence = field_extracted.get("confidence", 0.0)

        # line_items → exact-match each auto_search column per row
        if schema_field.type == "line_items":
            items = value if isinstance(value, list) else []
            li_auto_search = parsed_schema[field_name].get_line_item_auto_search_fields()
            resolved_rows: List[Dict[str, Any]] = []

            for row_idx, row in enumerate(items):
                resolved_row = dict(row)
                row_error_cols: List[str] = []

                # Check for empty/zero quantity field
                qty_val = None
                has_qty_field = False
                for qty_col in ("supplier_qty", "qty", "quantity"):
                    if qty_col in schema_field.item_fields or qty_col in resolved_row:
                        has_qty_field = True
                        q_val = resolved_row.get(qty_col)
                        if _is_zero_or_empty_qty(q_val):
                            qty_val = q_val if q_val is not None else ""
                            break

                if has_qty_field and qty_val is not None:
                    q_str = f" ('{qty_val}')" if qty_val != "" else ""
                    warnings_list.append(f"Row {row_idx + 1} removed from line items due to empty/zero quantity{q_str}")
                    continue

                for col, master_key in li_auto_search.items():
                    raw_val = resolved_row.get(col)
                    if isinstance(raw_val, dict):
                        raw_val = raw_val.get("label") or raw_val.get("value") or raw_val.get("name") or (str(raw_val.get("id")) if raw_val.get("id") is not None else None)

                    if _is_alt_unit_col(col) and (raw_val is None or str(raw_val).strip() == ""):
                        resolved_row[col] = {"id": None, "label": raw_val if raw_val is not None else ""}
                        continue

                    try:
                        m: MatchResult = match_entity_exact(
                            raw=raw_val,
                            master_key=master_key,
                            client_id=client_id,
                            parent_id=None,
                        )
                        resolved_row[col] = {
                            "id": m.matched_id,
                            "label": m.matched_label,
                            **m.extra_data,
                        }
                        if _is_alt_unit_col(col) and m.matched_id is not None:
                            alt_qty_col = _find_alt_qty_col(col, resolved_row, schema_field.item_fields)
                            if alt_qty_col:
                                alt_qty_val = resolved_row.get(alt_qty_col)
                                if isinstance(alt_qty_val, dict):
                                    alt_qty_val = alt_qty_val.get("value") or alt_qty_val.get("label")
                                if _is_zero_or_empty_qty(alt_qty_val):
                                    q_str = f" ('{alt_qty_val}')" if alt_qty_val not in (None, "") else ""
                                    alt_qty_title = alt_qty_col.replace("_", " ").title()
                                    warnings_list.append(f"Row {row_idx + 1}: {alt_qty_title}{q_str} must be greater than 0")
                    except EntityMatchError as exc:
                        col_title = col.replace("_", " ").title()
                        val_str = f" ('{raw_val}')" if raw_val else ""
                        if _is_alt_unit_col(col):
                            resolved_row[col] = {"id": None, "label": raw_val}
                            warnings_list.append(f"Row {row_idx + 1}: {col_title}{val_str} not found in master records")
                            logger.warning(
                                "Exact match failed for alt unit (row retained): field=%s row=%d col=%s raw='%s' | request_id=%s: %s",
                                field_name, row_idx + 1, col, raw_val, request_id, exc,
                            )
                        else:
                            row_error_cols.append(f"{col_title}{val_str}")
                            logger.warning(
                                "Exact match failed: field=%s row=%d col=%s raw='%s' | request_id=%s: %s",
                                field_name, row_idx + 1, col, raw_val, request_id, exc,
                            )

                if row_error_cols:
                    # Remove line item row if any quality/design/chart/color master match fails
                    err_summary = "<br>".join(row_error_cols)
                    warnings_list.append(f"Row {row_idx + 1} removed from line items due to invalid {err_summary}")
                    continue

                # Quantity normalisation & float rounding (2 decimal places)
                for qty_col in ("supplier_qty", "qty", "quantity"):
                    if qty_col in resolved_row and resolved_row[qty_col] is not None:
                        val = resolved_row[qty_col]
                        if isinstance(val, (int, float)) and val >= 1000 and val % 1000 == 0:
                            val = val / 1000.0
                        elif isinstance(val, str) and val.isdigit() and int(val) >= 1000 and int(val) % 1000 == 0:
                            val = int(val) / 1000.0

                        try:
                            f_val = float(val)
                            resolved_row[qty_col] = f"{round(f_val, 2):.2f}"
                        except (ValueError, TypeError):
                            pass

                resolved_rows.append(resolved_row)

            result_data[field_name] = LineItemsFieldResult(
                items=resolved_rows,
                confidence=confidence if resolved_rows else 0.0,
            )
            continue

        # direct → return extracted value as-is
        if schema_field.type == "direct":
            result_data[field_name] = DirectFieldResult(
                value=value,
                confidence=confidence if value is not None else 0.0,
            )
            continue

        # auto_search → exact match only
        parent_id: int | None = None
        if schema_field.parent_field and schema_field.parent_field in matched_ids:
            parent_id = matched_ids[schema_field.parent_field]

        try:
            match: MatchResult = match_entity_exact(
                raw=value,
                master_key=schema_field.master,  # type: ignore[arg-type]
                client_id=client_id,
                parent_id=parent_id,
            )
            matched_ids[field_name] = match.matched_id
            result_data[field_name] = AutoSearchFieldResult(
                master=schema_field.master or "",
                raw=match.raw,
                matched_id=match.matched_id,
                matched_label=match.matched_label,
                confidence=match.confidence,
                alternatives=[],
                parent_field=schema_field.parent_field,
                extra_data=match.extra_data,
            )
        except EntityMatchError as exc:
            field_title = field_name.replace("_", " ").title()
            val_str = f" ('{value}')" if value else ""
            warnings_list.append(f"{field_title}{val_str} not found in master records")
            matched_ids[field_name] = None
            result_data[field_name] = AutoSearchFieldResult(
                master=schema_field.master or "",
                raw=value,
                matched_id=None,
                matched_label=None,
                confidence=0.0,
                alternatives=[],
                parent_field=schema_field.parent_field,
            )
            logger.warning(
                "Exact match failed: field='%s' master='%s' raw='%s' | request_id=%s: %s",
                field_name, schema_field.master, value, request_id, exc,
            )

    # ── Assemble sectioned response ────────────────────────────────────────────
    processing_ms = int((time.monotonic() - start_time) * 1000)

    basic_details: Dict[str, Any] = {}
    line_items_out: List[Dict[str, Any]] = []
    summary_out: Dict[str, Any] = {}

    for field_name, field_result in result_data.items():
        schema_field = parsed_schema[field_name]

        if isinstance(field_result, LineItemsFieldResult):
            line_items_out = field_result.items
            continue

        if isinstance(field_result, DirectFieldResult):
            plain_value: Any = field_result.value
        elif isinstance(field_result, AutoSearchFieldResult):
            if field_result.matched_id is not None:
                plain_value = {
                    "id": field_result.matched_id,
                    "label": field_result.matched_label,
                    **field_result.extra_data,
                }
            else:
                raw_val = field_result.raw
                if isinstance(raw_val, dict):
                    raw_val = raw_val.get("label") or raw_val.get("value") or raw_val.get("name") or (str(raw_val.get("id")) if raw_val.get("id") is not None else None)
                plain_value = raw_val if raw_val is not None else ""
        elif isinstance(field_result, AutoPopulatedFieldResult):
            plain_value = None
        else:
            plain_value = None

        if schema_field.section == "summary":
            summary_out[field_name] = plain_value
        else:
            basic_details[field_name] = plain_value

    warning_message: str | None = None
    if warnings_list:
        warning_message = "<br>".join(warnings_list)

    logger.info(
        json.dumps({
            "request_id": request_id,
            "client_id": client_id,
            "processing_time_ms": processing_ms,
            "status": "success",
            "warnings_count": len(warnings_list),
            "endpoint": "/api/v1/invoice/process",
        })
    )

    return SectionedInvoiceParseResponse(
        status="success",
        basic_details=basic_details,
        line_items=line_items_out,
        summary=summary_out,
        warning_message=warning_message,
        meta=ResponseMeta(
            model=settings.GEMINI_PARSE_MODEL,
            pages=0,          # no file was uploaded at this step
            scan_cost_usd=0.0,  # cost was already charged in /extract
            processed_at=datetime.now(tz=timezone.utc),
        ),
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="Health check")
@app.get("/health/", response_model=HealthResponse, include_in_schema=False)
@app.get("/ocr-api/health", response_model=HealthResponse, include_in_schema=False)
@app.get("/ocr-api/health/", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns basic service health including entity index status.
    """
    return HealthResponse(
        status="ok",
        index_loaded=indexer.is_loaded(),
        indexed_clients=indexer.indexed_client_count(),
        uptime_seconds=round(time.monotonic() - _startup_time, 2),
    )


# ── Manual index refresh ──────────────────────────────────────────────────────

@app.get(
    "/api/v1/index/refresh",
    response_model=IndexRefreshResponse,
    summary="Refresh entity index",
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/api/v1/index/refresh",
    response_model=IndexRefreshResponse,
    summary="Refresh entity index",
    dependencies=[Depends(verify_api_key)],
)
@app.get(
    "/api/v1/master/refresh",
    response_model=IndexRefreshResponse,
    summary="Refresh entity index (alias)",
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/api/v1/master/refresh",
    response_model=IndexRefreshResponse,
    summary="Refresh entity index (alias)",
    dependencies=[Depends(verify_api_key)],
)
async def refresh_index() -> IndexRefreshResponse:
    """
    Manually trigger an entity index reload from MySQL.

    Useful after bulk master-data changes in the ERP without waiting for
    the scheduled TTL refresh.
    """
    logger.info("Manual entity index refresh triggered.")
    try:
        await indexer.load()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Manual index refresh failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Index refresh failed: {exc}",
        )

    client_count = indexer.indexed_client_count()
    return IndexRefreshResponse(
        status="refreshed",
        indexed_clients=client_count,
        message=(
            f"Entity index refreshed successfully. "
            f"{client_count} client(s) indexed."
        ),
    )
