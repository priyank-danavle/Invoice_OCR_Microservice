"""
app/ocr/parser.py
─────────────────
Uses Gemini Flash in JSON mode to extract field values from raw
OCR text, guided entirely by the output_schema supplied by the ERP backend.

Uses google-genai SDK (import path: google.genai).

No field names are hardcoded here. The parser is schema-driven:
  - "direct" fields      → extract value + confidence from OCR text
  - "auto_search" fields → extract the raw entity name string only
                           (fuzzy matching happens in matcher.py)
  - "auto_populated" fields → skipped entirely
  - "line_items" fields  → extract a list of dicts, one per table row,
                           using the columns declared in item_fields

Textile-specific extraction hints are embedded in the system prompt to
improve accuracy for HSN codes, Indian dates, units, and number formats.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import settings
from app.models.request import OutputSchema
from app.ocr.challan_validator import check_and_raise_single_challan
from app.ocr.inward_classifier import determine_inward_type_from_dict

logger = logging.getLogger(__name__)

# ── Custom exceptions ─────────────────────────────────────────────────────────

class ParserError(Exception):
    """Raised when field extraction fails or the model returns invalid JSON."""


# ── System prompt (static, textile-domain hints) ──────────────────────────────

_SYSTEM_PROMPT = """You are a specialist data extraction engine for textile industry invoices and purchase orders in India.

Extract field values from OCR text with the following domain knowledge:

TEXTILE DOMAIN RULES:
- HSN codes are 4–8 digit numeric codes (chapters 50–63 cover textiles, e.g. 5208, 5209, 5512, 6006)
- Units: MTR (metres), KG (kilograms), PCS (pieces), YDS (yards), SET, ROLL
- Indian date formats DD/MM/YYYY or DD-MM-YYYY → always normalise to YYYY-MM-DD
- GST rates applicable to textiles: 0%, 5%, 12%, 18% — extract only the numeric percentage
- Indian number format: 1,00,000 = 100000 (lakh notation) — always return without commas
- Decimal quantities: In textile documents, floating-point quantities and amounts must be rounded up to 2 decimal places (e.g. "20.00", "6.67", "14.29", "88.19"). A dot (.) is ALWAYS a decimal point, NEVER a thousands separator.
- Currency values may include ₹ or Rs. prefix — strip the symbol, return numeric string only
- Discount and brokerage may be expressed as percentages (e.g. "2%") or flat amounts

LINE ITEM EXTRACTION RULES:
- Include ONLY individual physical product/item detail rows in line_items.
- STRICTLY EXCLUDE table headers, summary rows, total rows, subtotals, grand totals, net totals, tax totals, total quantity/roll counts, remarks, terms, or declaration rows.
- Rows containing label strings such as "Total", "Grand Total", "Sub Total", "Subtotal", "Net Amount", "Total Qty", "Total Quantity", "Cartons", etc. are summary rows and MUST NOT be extracted into line_items.

COLUMN ALIAS MAPPINGS FOR LINE ITEMS:
- "supplier_qty": extract numeric quantities from table columns named "Qty", "Quantity", "Weight", "Wts", "Mtrs", "Pcs/Qty", "Net Qty", "Gross Qty"
- "primary_unit": extract unit string from table columns named "Unit", "UOM", "Per"
- "bale_number": extract identifier from table columns named "Bale No", "Roll No", "Pcs No", "Lot No", "Carton No"
- "supplier_quality": extract item description from table columns named "Description of Goods", "Particulars", "Quality", "Item Name"
- "width": extract width from table columns named "Width", "Width (in)", "Size"
- "gsm": extract gsm from table columns named "Gsm", "GSM", "Weight/GSM"

CONFIDENCE SCORING:
- 0.90–0.99: Value is clearly and unambiguously present in the document
- 0.70–0.89: Value is present but in an unusual or abbreviated format
- 0.50–0.69: Value is partially visible or must be inferred from context
- null: Field is NOT present in the document — NEVER guess or invent a value

OUTPUT RULES:
- Return ONLY a valid JSON object, with no markdown fences, no preamble, no explanation
- Use null (not "null", not "N/A") when a field is not found
- Confidence is a float in [0.0, 1.0]; use null confidence when value is null
- Never hallucinate values — if uncertain at any confidence level below 0.50, return null
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_unified_prompt(
    raw_text: str,
    fields_to_extract: List[Tuple[str, str]],
    line_item_fields: List[Tuple[str, Dict[str, str]]],
) -> str:
    """
    Build a single unified extraction prompt for both direct/auto_search fields
    AND line_item table fields in one LLM call.
    """
    desc_lines: List[str] = []
    structure_lines: List[str] = []

    # 1. Direct / Auto_search fields
    if fields_to_extract:
        for name, hint in fields_to_extract:
            desc = "raw numeric or text value" if hint == "direct_value" else "entity name as it appears in the document (company/person/place name)"
            desc_lines.append(f'- "{name}": extract the {desc}')
            structure_lines.append(f'  "{name}": {{"value": <extracted_value_or_null>, "confidence": <float_or_null>}}')

    # 2. Line Items fields
    if line_item_fields:
        for li_field_name, item_fields in line_item_fields:
            cols = ", ".join(f'"{col}": <value_or_null>' for col in item_fields)
            desc_lines.append(
                f'- "{li_field_name}": extract array of table rows from the item/dispatch table with columns ({", ".join(item_fields.keys())}). Note: column values inside row objects MUST be plain scalar strings/numerics/null directly, NOT nested {{"value": ..., "confidence": ...}} objects.'
            )
            structure_lines.append(
                f'  "{li_field_name}": {{\n'
                f'    "value": [{{\n      {cols}\n    }}],\n'
                f'    "confidence": <float 0.0–1.0>\n'
                f'  }}'
            )

    field_descriptions = "\n".join(desc_lines)
    json_structure = ",\n".join(structure_lines)

    return f"""Extract the following fields from this OCR document text.

FIELDS TO EXTRACT:
{field_descriptions}

Rules:
- For line item tables: Return one dict per physical product item row in the table.
- Column mapping: "supplier_qty" MUST be extracted from columns named "Qty", "Quantity", or "Weight". "primary_unit" from "Unit" or "UOM". "bale_number" from "Roll No", "Bale No", or "Pcs No".
- STRICTLY EXCLUDE summary/total rows (e.g. rows labeled "Total", "Grand Total", "Sub Total", "Net Amount", "Total Qty", etc.), header rows, remarks, and footers. Do NOT add total rows as line items.
- Numbers must be plain numerics without commas or units. Convert any floating-point numbers to 2 decimal places (e.g. 20.00, 6.67, 14.29, 88.19).
- Normalise dates to YYYY-MM-DD. Strip currency symbols from amounts.

DOCUMENT TEXT:
---
{raw_text}
---

Return a JSON object with exactly this structure:
{{
{json_structure}
}}"""


# ── Response normaliser ───────────────────────────────────────────────────────

def _normalise_date(value: str) -> str:
    """Normalise Indian date formats to ISO 8601 (YYYY-MM-DD)."""
    pattern = re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$")
    match = pattern.match(value.strip())
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return value


def _clean_value(field_name: str, value: Any) -> Optional[Any]:
    """
    Post-process an extracted value.

    - Passes list values through unchanged (used by line_items rows).
    - Unwraps dict objects with a 'value', 'label', 'name', or 'raw' key.
    - Strips currency symbols (₹, Rs.)
    - Removes commas from Indian-format numbers
    - Converts floating-point numbers to 2 decimal places
    - Normalises date fields
    - Returns None if value is None or empty string / 'None' / 'null'
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if "value" in value:
            value = value.get("value")
        elif "label" in value:
            value = value.get("label")
        elif "name" in value:
            value = value.get("name")
        elif "raw" in value:
            value = value.get("raw")
        else:
            value = None
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in ("none", "null"):
        return None

    text = re.sub(r"^(₹|Rs\.)\s*", "", text).strip()

    if re.match(r"^[\d,]+(\.\d+)?$", text):
        text = text.replace(",", "")

    # Convert floating-point numbers to 2 decimal places
    if re.match(r"^-?\d+\.\d+$", text):
        try:
            val_float = round(float(text), 2)
            text = f"{val_float:.2f}"
        except ValueError:
            pass

    if "date" in field_name.lower():
        text = _normalise_date(text)

    return text if text else None


_SUMMARY_KEYWORDS_PATTERN = re.compile(
    r"^\s*(total|grand\s+total|sub\s*total|net\s+total|net\s+amount|total\s+(qty|quantity|pcs|pieces|rolls|cartons)|total\s+amount)\b",
    re.IGNORECASE,
)


def _is_zero_or_empty_qty(val: Any) -> bool:
    """Check if quantity value is None, empty, zero, or dash."""
    if val is None:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    val_str = str(val).strip()
    if not val_str or val_str in ("-", "null", "None"):
        return True
    try:
        if float(val_str) == 0:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _is_summary_or_header_row(row: Dict[str, Any]) -> bool:
    """
    Check whether a parsed table row is actually a summary/total row,
    a header row, or an empty row that should be excluded from line_items.
    """
    non_empty_values = [v for v in row.values() if v is not None and str(v).strip() != ""]
    if not non_empty_values:
        return True

    for val in row.values():
        if val is None:
            continue
        val_str = str(val).strip()
        if _SUMMARY_KEYWORDS_PATTERN.search(val_str):
            return True

    return False


# ── Model Call Helper with Fallback ───────────────────────────────────────────


async def _generate_content_with_fallback(
    client: genai.Client,
    prompt: str,
    config: types.GenerateContentConfig,
) -> Any:
    """
    Execute generate_content using settings.GEMINI_PARSE_MODEL.
    If a 404 or server error occurs, falls back automatically to alternative models.
    """
    models_to_try: List[str] = [
        settings.GEMINI_PARSE_MODEL,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]
    unique_models: List[str] = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    last_exc: Exception | None = None
    for idx, model_name in enumerate(unique_models):
        try:
            return await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        except APIError as exc:
            code = getattr(exc, "code", getattr(exc, "status_code", 0))
            if code in (404, 429, 500, 502, 503, 504) and idx < len(unique_models) - 1:
                logger.warning(
                    "Gemini model '%s' returned HTTP %s. Retrying with fallback model '%s'...",
                    model_name, code, unique_models[idx + 1],
                )
                last_exc = exc
                if code in (429, 503, 504):
                    await asyncio.sleep(1.0)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    if last_exc:
        raise last_exc


# ── Public API ────────────────────────────────────────────────────────────────

async def parse(
    raw_text: str,
    output_schema: OutputSchema,
    client_id: Optional[int] = None,
    branch_gst_list: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract field values from OCR text guided by the output_schema in a single
    unified Gemini call.

    Args:
        raw_text: Full concatenated OCR text from all pages.
        output_schema: Validated schema dict from the ERP backend.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of field_name →
            {"value": ..., "confidence": float}
            For line_items fields, value is a List[Dict].

    Raises:
        ParserError: If the model returns invalid JSON or parsing fails after retries.
    """
    fields_to_extract: List[Tuple[str, str]] = []       # (name, hint) for direct/auto_search
    line_item_fields: List[Tuple[str, Dict[str, str]]] = []  # (name, item_fields) for line_items

    for field_name, schema_field in output_schema.items():
        if schema_field.type == "auto_populated" and field_name != "inward_type":
            continue
        if schema_field.type == "line_items":
            line_item_fields.append((field_name, schema_field.item_fields or {}))
            continue
        hint = "direct_value" if schema_field.type in ("direct", "auto_populated") else "entity_name_string"
        fields_to_extract.append((field_name, hint))

    has_inward_type = "inward_type" in output_schema
    helper_fields_added: List[str] = []
    if has_inward_type:
        helper_candidates: List[Tuple[str, str]] = [
            ("buyer_name", "entity_name_string"),
            ("buyer_address", "direct_value"),
            ("buyer_pan", "direct_value"),
            ("buyer_gst", "direct_value"),
            ("consignee_name", "entity_name_string"),
            ("consignee_address", "direct_value"),
            ("consignee_pan", "direct_value"),
            ("consignee_gst", "direct_value"),
        ]
        existing_names = {name for name, _ in fields_to_extract}
        for name, hint in helper_candidates:
            if name not in existing_names and name not in output_schema:
                fields_to_extract.append((name, hint))
                helper_fields_added.append(name)

    if not fields_to_extract and not line_item_fields:
        logger.debug("No extractable fields in schema — nothing to parse.")
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    max_attempts = 4
    base_delay = 5.0

    gen_config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )

    prompt = _build_unified_prompt(raw_text, fields_to_extract, line_item_fields)

    logger.info(
        "Sending unified Gemini parse request (%d direct fields, %d line_item fields).\n"
        "--- GEMINI PARSE PAYLOAD START ---\n"
        "[System Instruction]\n%s\n\n"
        "[User Prompt]\n%s\n"
        "--- GEMINI PARSE PAYLOAD END ---",
        len(fields_to_extract), len(line_item_fields), _SYSTEM_PROMPT, prompt,
    )

    last_exc: Exception | None = None
    extracted: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug("Gemini parse call attempt %d/%d.", attempt, max_attempts)
            start = time.monotonic()
            response = await _generate_content_with_fallback(
                client=client,
                prompt=prompt,
                config=gen_config,
            )

            # ── Gemini parser token/cost usage ──────────────────────────────
            usage = response.usage_metadata

            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0

            INPUT_PRICE_PER_1M = 0.30
            OUTPUT_PRICE_PER_1M = 2.50

            input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
            output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
            parser_cost = input_cost + output_cost

            logger.info(
                "Gemini Parser usage: "
                "input_tokens=%d, output_tokens=%d, total_tokens=%d, "
                "input_cost=$%.8f, output_cost=$%.8f, estimated_cost=$%.8f",
                input_tokens,
                output_tokens,
                total_tokens,
                input_cost,
                output_cost,
                parser_cost,
            )

            elapsed = time.monotonic() - start
            raw_json: str = response.text or ""
            logger.info(
                "Gemini parse responded in %.2f s.\n"
                "--- GEMINI PARSE RESPONSE START ---\n%s\n"
                "--- GEMINI PARSE RESPONSE END ---",
                elapsed, raw_json,
            )
            extracted = json.loads(raw_json)
            break

        except json.JSONDecodeError as exc:
            logger.error("Parser JSON decode error (attempt %d): %s", attempt, exc)
            last_exc = exc
            if attempt == max_attempts:
                raise ParserError(
                    f"Gemini returned invalid JSON after {max_attempts} attempts: {exc}"
                ) from exc
            await asyncio.sleep(base_delay * attempt)

        except APIError as exc:
            code = getattr(exc, "code", getattr(exc, "status_code", 0))
            msg = str(exc)
            if code in (429, 500, 502, 503, 504):
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                if match:
                    delay = float(match.group(1)) + 1.0
                else:
                    delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini parse API rate-limited/error (status=%s). Retrying in %.1f s … (attempt %d/%d)",
                    code, delay, attempt, max_attempts,
                )
                last_exc = exc
                await asyncio.sleep(delay)
                continue
            raise ParserError(f"Gemini API error (code={code}): {exc}") from exc

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unexpected parser error: %s", exc)
            raise ParserError(f"Unexpected error during field parsing: {exc}") from exc

    else:
        raise ParserError(
            f"Field parsing failed after {max_attempts} attempts. Last error: {last_exc}"
        )

    # ── Post-process direct & auto_search results ─────────────────────────────
    for field_name, _ in fields_to_extract:
        raw_entry = extracted.get(field_name, {}) or {}
        if isinstance(raw_entry, dict):
            raw_value = raw_entry.get("value")
            raw_conf = raw_entry.get("confidence")
        else:
            raw_value = raw_entry
            raw_conf = 0.8

        cleaned = _clean_value(field_name, raw_value)
        confidence: float = 0.0
        if cleaned is not None:
            if raw_conf is not None:
                try:
                    confidence = float(raw_conf)
                    confidence = max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    confidence = 0.8
            else:
                confidence = 0.8

        results[field_name] = {"value": cleaned, "confidence": confidence}

    # ── Post-process line_items results ───────────────────────────────────────
    for li_field_name, item_fields in line_item_fields:
        raw_entry = extracted.get(li_field_name, {}) or {}
        if isinstance(raw_entry, dict):
            raw_value = raw_entry.get("value")
            raw_conf = raw_entry.get("confidence")
        elif isinstance(raw_entry, list):
            raw_value = raw_entry
            raw_conf = 0.95
        else:
            raw_value = []
            raw_conf = 0.0

        if isinstance(raw_value, list):
            cleaned_rows: List[Dict[str, Any]] = []
            for row in raw_value:
                if isinstance(row, dict):
                    cleaned_row: Dict[str, Any] = {}
                    for col in item_fields.keys():
                        val = row.get(col)
                        if val is None:
                            if col == "supplier_qty":
                                val = row.get("qty") or row.get("quantity") or row.get("weight") or row.get("net_qty") or row.get("pcs_qty")
                            elif col == "primary_unit":
                                val = row.get("unit") or row.get("uom")
                            elif col == "bale_number":
                                val = row.get("roll_no") or row.get("bale_no") or row.get("pcs_no") or row.get("roll_number")
                            elif col == "supplier_quality":
                                val = row.get("description_of_goods") or row.get("particulars") or row.get("quality")
                        cleaned_row[col] = _clean_value(col, val)
                    if _is_summary_or_header_row(cleaned_row):
                        continue
                    if any(k in item_fields.keys() for k in ("supplier_qty", "qty", "quantity")):
                        q_val = cleaned_row.get("supplier_qty") or cleaned_row.get("qty") or cleaned_row.get("quantity")
                        if _is_zero_or_empty_qty(q_val):
                            continue
                    cleaned_rows.append(cleaned_row)

            confidence = 0.95
            if raw_conf is not None:
                try:
                    confidence = max(0.0, min(1.0, float(raw_conf)))
                except (TypeError, ValueError):
                    confidence = 0.95

            results[li_field_name] = {"value": cleaned_rows, "confidence": confidence}
        else:
            results[li_field_name] = {"value": [], "confidence": 0.0}

    if has_inward_type:
        inward_enum = determine_inward_type_from_dict(
            results,
            branch_gst_list=branch_gst_list,
            client_id=client_id,
        )
        results["inward_type"] = {"value": inward_enum.value, "confidence": 1.0}
        for h_field in helper_fields_added:
            results.pop(h_field, None)

    # ── Single delivery challan validation ───────────────────────────────────
    check_and_raise_single_challan(raw_text, results)

    logger.info(
        "Parser extracted %d field(s) (%d line_item field(s)).",
        len(results), len(line_item_fields),
    )
    return results
