"""
app/ocr/challan_validator.py
──────────────────────────────
Validation logic to enforce single delivery challan uploads per request.

Every single delivery challan has only ONE Buyer (Bill To) section and ONE
Consignee (Ship To) section across all its pages.
If a document or batch of pages contains multiple distinct Buyer or Consignee
header sections, or multiple distinct Challan Numbers with separate header blocks,
it indicates that multiple distinct delivery challans were uploaded together.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MultipleChallansDetectedError(Exception):
    """Raised when an uploaded document contains multiple distinct delivery challans."""

    def __init__(self, message: str = "Multiple delivery challans detected in the uploaded document. Every request must contain only a single delivery challan (which may span multiple pages). Please upload pages belonging to one delivery challan at a time.") -> None:
        super().__init__(message)
        self.message = message


def _clean_str(s: Optional[Any]) -> str:
    """Normalize string for detail comparison."""
    if not s:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(cleaned.split())


def _clean_header_value(val: str) -> str:
    """
    Strip table pipes, inline field labels, dates, and extraneous text from header values.
    For example: 'BFSD2627-00017 (21/09/2026) | Private Mark : ...' -> 'bfsd2627 00017'
    """
    if not val:
        return ""
    # Stop at pipe character '|' if present
    if "|" in val:
        val = val.split("|")[0]
    # Stop at inline labels if concatenated on same line
    val = re.split(r"\b(?:date|private mark|order no|order|pan|gst|gstin|phone|state|no\. of|lr|vehicle)\b", val, flags=re.IGNORECASE)[0]
    return _clean_str(val)


def _extract_header_occurrences(raw_text: str, label_pattern: str) -> List[Tuple[str, str]]:
    """
    Find non-empty string values following header labels (e.g. Buyer:, Consignee:, Bill To:).
    Returns list of tuples: (raw_match, normalized_clean_string).
    """
    if not raw_text:
        return []

    found: List[Tuple[str, str]] = []
    seen_clean: List[str] = []
    # Match label followed by value up to pipe |, newline, or tab
    matches = re.findall(rf"(?:{label_pattern})\s*[:\-]?\s*([^|\n\r\t]+)", raw_text, re.IGNORECASE)
    for m in matches:
        raw_val = m.strip()
        cleaned = _clean_header_value(raw_val)
        if len(cleaned) > 2 and cleaned not in seen_clean:
            seen_clean.append(cleaned)
            found.append((raw_val, cleaned))
    return found


def _are_substantially_different(s1: str, s2: str) -> bool:
    """
    Return True if s1 and s2 represent completely different entity/challan names.
    Return False if they are identical, substrings, or share substantial word overlap.
    """
    c1, c2 = _clean_str(s1), _clean_str(s2)
    if not c1 or not c2:
        return False
    if c1 == c2 or c1 in c2 or c2 in c1:
        return False

    # Check for distinct numeric identifiers inside strings (e.g. DC-8801 vs DC-9909)
    nums1 = set(re.findall(r"\d+", c1))
    nums2 = set(re.findall(r"\d+", c2))
    if nums1 and nums2 and nums1 != nums2:
        return True

    words1 = set(c1.split())
    words2 = set(c2.split())
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    overlap_ratio = len(intersection) / min(len(words1), len(words2))
    if overlap_ratio > 0.5:
        return False
    return True


def validate_single_challan(
    raw_text: str,
    parsed_values: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that raw OCR text and parsed values represent a single delivery challan.

    Single challan criteria:
    - Multi-page single challans are allowed (share the same Buyer / Consignee / Challan No).
    - If there are multiple distinct Buyer headers or Consignee headers referring to
      different parties across pages, or multiple distinct Challan Numbers found,
      it indicates multiple separate challans were concatenated.

    Args:
        raw_text: Combined raw text from OCR.
        parsed_values: Optional dictionary of extracted fields.

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not raw_text:
        return True, None

    # Check 1: Inspect parsed_values metadata if LLM explicitly flagged multiple challans
    if parsed_values and isinstance(parsed_values, dict):
        flag = parsed_values.get("multiple_challans_detected")
        if isinstance(flag, dict):
            flag = flag.get("value")
        if flag is True or str(flag).lower() in ("true", "1", "yes"):
            msg = (
                "Multiple delivery challans detected in the uploaded document. "
                "Every request must contain only a single delivery challan (which may span multiple pages). "
                "Please upload pages belonging to one delivery challan at a time."
            )
            return False, msg

    buyer_patterns = r"buyer|bill\s+to|customer|party\s+name"
    consignee_patterns = r"consignee|ship\s+to|delivery\s+at|delivery\s+to"
    challan_no_patterns = r"challan\s+no|challan\s+number|dc\s+no|invoice\s+no|invoice\s+number"

    # Check 2: Scan for multi-file sections or multi-page blocks
    file_sections = [s.strip() for s in re.split(r"===\s*File:\s*[^\n]+===", raw_text) if s.strip()]

    if len(file_sections) > 1:
        section_buyers: List[Tuple[str, str]] = []
        section_consignees: List[Tuple[str, str]] = []

        for sec in file_sections:
            b = _extract_header_occurrences(sec, buyer_patterns)
            c = _extract_header_occurrences(sec, consignee_patterns)
            if b:
                section_buyers.extend(b)
            if c:
                section_consignees.extend(c)

        # Check for multiple distinct buyer names across files
        if len(section_buyers) >= 2:
            first_raw, first_clean = section_buyers[0]
            different = [raw for raw, clean in section_buyers[1:] if _are_substantially_different(clean, first_clean)]
            if different:
                msg = (
                    f"Multiple delivery challans detected across uploaded files (Buyer: '{first_raw}' vs '{different[0]}'). "
                    "Please upload pages for a single delivery challan only."
                )
                logger.warning("Single-challan validation failed: different buyers across files (%s vs %s)", first_raw, different[0])
                return False, msg

        # Check for multiple distinct consignee names across files
        if len(section_consignees) >= 2:
            first_raw, first_clean = section_consignees[0]
            different = [raw for raw, clean in section_consignees[1:] if _are_substantially_different(clean, first_clean)]
            if different:
                msg = (
                    f"Multiple delivery challans detected across uploaded files (Consignee: '{first_raw}' vs '{different[0]}'). "
                    "Please upload pages for a single delivery challan only."
                )
                logger.warning("Single-challan validation failed: different consignees across files (%s vs %s)", first_raw, different[0])
                return False, msg

    # Check 3: Check for multiple distinctly different Challan numbers in overall text
    distinct_challan_nos = _extract_header_occurrences(raw_text, challan_no_patterns)
    if len(distinct_challan_nos) > 1:
        first_raw, first_clean = distinct_challan_nos[0]
        different = [raw for raw, clean in distinct_challan_nos[1:] if _are_substantially_different(clean, first_clean)]
        if different:
            msg = (
                f"Multiple delivery challans detected in the document (Challan Nos: '{first_raw}', '{different[0]}'). "
                "Please upload only one delivery challan per request."
            )
            logger.warning("Single-challan validation failed: multiple challan numbers found (%s, %s)", first_raw, different[0])
            return False, msg

    return True, None


def check_and_raise_single_challan(
    raw_text: str,
    parsed_values: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run single-challan validation and raise MultipleChallansDetectedError if invalid.
    """
    is_valid, err_msg = validate_single_challan(raw_text, parsed_values)
    if not is_valid:
        raise MultipleChallansDetectedError(err_msg or "Multiple delivery challans detected in uploaded document.")
