"""
app/models/request.py
─────────────────────
Pydantic models for incoming request validation.
The output_schema is received as a raw JSON string from the ERP backend
and is validated here before any processing begins.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Schema field type literals ────────────────────────────────────────────────

FieldType = Literal["direct", "auto_search", "auto_populated", "line_items"]


class SchemaField(BaseModel):
    """
    Definition of a single field inside the output_schema.

    Attributes:
        type: Extraction strategy — direct, auto_search, auto_populated,
              or line_items.
        master: Required when type == "auto_search". Key into the entity index.
        parent_field: Optional. For dependent lookups (e.g. delivery_address
                      depends on supplier_name being matched first).
        item_fields: Required when type == "line_items". Dict mapping each
                     column name to its extraction hint string, e.g.
                     {"roll_no": "text", "qty": "number", "unit": "text"}.
                     The LLM uses these keys to build a list of dicts, one per
                     row found in the document's dispatch/item table.
        section: Response grouping key. One of "basic_details", "line_items",
                 or "summary". Defaults to "basic_details".
                 Fields with type "line_items" are always placed in the
                 "line_items" section regardless of this value.
    """

    type: FieldType
    master: Optional[str] = Field(
        default=None,
        description="Entity index key; required for auto_search fields.",
    )
    parent_field: Optional[str] = Field(
        default=None,
        description="Field that must be resolved before this one (dependency).",
    )
    item_fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Column definitions for line_items rows. "
            "String value = direct extraction hint ('text', 'number', 'date'). "
            "Dict value = auto_search column, e.g. {'master': 'quality'}."
        ),
    )
    section: str = Field(
        default="basic_details",
        description=(
            "Response section this field belongs to. "
            "One of: 'basic_details', 'line_items', 'summary'. "
            "Fields with type='line_items' are always placed in 'line_items'."
        ),
    )

    @model_validator(mode="after")
    def validate_field_type_constraints(self) -> "SchemaField":
        """
        Cross-field validation:
          - auto_search fields must have a master key.
          - line_items fields must have item_fields defined.
          - section must be one of the three allowed values.

        Raises:
            ValueError: On constraint violations.
        """
        if self.type == "auto_search" and not self.master:
            raise ValueError(
                "auto_search fields must specify a 'master' key "
                "(e.g. 'supplier', 'transport')."
            )
        if self.type == "line_items" and not self.item_fields:
            raise ValueError(
                "line_items fields must specify 'item_fields' "
                "(e.g. {\"roll_no\": \"text\", \"qty\": \"number\"})."
            )
        allowed_sections = {"basic_details", "line_items", "summary"}
        if self.section not in allowed_sections:
            raise ValueError(
                f"section must be one of {sorted(allowed_sections)}, got '{self.section}'."
            )
        return self

    def get_line_item_auto_search_fields(self) -> Dict[str, str]:
        """
        Return a mapping of {column_name: master_key} for all auto_search
        columns declared in item_fields.

        Used by the caller to run fuzzy matching on extracted row values.

        Returns:
            Dict[str, str]: e.g. {"quality_master": "quality",
                                   "design_master": "design",
                                   "color_master": "color",
                                   "chart_master": "chart"}
        """
        if not self.item_fields:
            return {}
        return {
            col: cfg["master"]
            for col, cfg in self.item_fields.items()
            if isinstance(cfg, dict) and "master" in cfg
        }



# ── Parsed output schema type alias ──────────────────────────────────────────

OutputSchema = Dict[str, SchemaField]


def parse_output_schema(raw_json: str) -> OutputSchema:
    """
    Parse and validate the output_schema JSON string sent by the ERP backend.

    Args:
        raw_json: JSON string representing the output schema.

    Returns:
        OutputSchema: Validated dict mapping field names to SchemaField objects.

    Raises:
        SchemaValidationError: If the JSON is malformed or schema fields are
                               invalid (imported at call site to avoid circular
                               imports — caller handles this exception).
        json.JSONDecodeError: If raw_json is not valid JSON.
        ValueError: If any schema field fails Pydantic validation.
    """
    data: Dict[str, Any] = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("output_schema must be a JSON object (dict), not a list.")
    return {field_name: SchemaField(**field_def) for field_name, field_def in data.items()}


class InvoiceParseFormData(BaseModel):
    """
    Represents the validated non-file fields of the multipart/form-data request.

    The file field is handled separately by FastAPI's UploadFile mechanism.

    Attributes:
        client_id: Scopes all entity lookups to this ERP client.
        output_schema_raw: Raw JSON string defining what to extract.
    """

    client_id: int = Field(..., gt=0, description="ERP client identifier (> 0).")
    output_schema_raw: str = Field(
        ...,
        alias="output_schema",
        description="JSON string defining schema fields.",
    )

    model_config = {"populate_by_name": True}
