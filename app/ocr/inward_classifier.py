"""
app/ocr/inward_classifier.py
──────────────────────────────
Classification logic to identify inward type based on buyer and consignee details.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.models.enums import InwardType


def extract_pan(val: Optional[Any]) -> Optional[str]:
    """
    Extract a 10-character Indian PAN string from a PAN or GSTIN string.

    Indian PAN format: 5 letters + 4 digits + 1 letter (e.g. AABCG0675P).
    Indian GSTIN format: 2 digits + 10 chars PAN + 3 chars (e.g. 24AABCG0675P1ZD).
    """
    if not val:
        return None
    val_str = str(val).strip().upper()
    if not val_str:
        return None

    # First, look for a 15-character GSTIN pattern and extract chars 3..12 (10-digit PAN)
    gst_match = re.search(r"\b\d{2}([A-Z]{5}\d{4}[A-Z])[A-Z0-9]{3}\b", val_str)
    if gst_match:
        return gst_match.group(1)

    # Next, look for a 10-character PAN pattern (5 uppercase letters, 4 digits, 1 uppercase letter)
    pan_match = re.search(r"[A-Z]{5}\d{4}[A-Z]", val_str)
    if pan_match:
        return pan_match.group(0)

def _normalize_str(s: Optional[Any]) -> str:
    """Normalize string for detail comparison."""
    if not s:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(cleaned.split())


def _normalize_gst(val: Optional[Any]) -> Optional[str]:
    """
    Normalize GST string for comparison.
    Extracts 15-character GSTIN if found, or removes non-alphanumeric characters.
    """
    if not val:
        return None
    val_str = str(val).strip().upper()
    if not val_str:
        return None
    gst_match = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]{3}\b", val_str)
    if gst_match:
        return gst_match.group(0)
    cleaned = re.sub(r"[^\w]", "", val_str)
    return cleaned if cleaned else None


def is_identical_buyer_consignee(
    buyer_name: Optional[str],
    buyer_address: Optional[str],
    buyer_pan: Optional[str],
    buyer_gst: Optional[str],
    consignee_name: Optional[str],
    consignee_address: Optional[str],
    consignee_pan: Optional[str],
    consignee_gst: Optional[str],
) -> bool:
    """
    Check if buyer and consignee details are identical.
    """
    b_name = _normalize_str(buyer_name)
    c_name = _normalize_str(consignee_name)
    b_addr = _normalize_str(buyer_address)
    c_addr = _normalize_str(consignee_address)

    b_pan_code = extract_pan(buyer_pan) or extract_pan(buyer_gst)
    c_pan_code = extract_pan(consignee_pan) or extract_pan(consignee_gst)

    # If neither buyer nor consignee information is present, not identical
    if not b_name and not c_name and not b_addr and not c_addr and not b_pan_code and not c_pan_code:
        return False

    names_match = False
    if b_name and c_name:
        if b_name == c_name or b_name in c_name or c_name in b_name:
            names_match = True
    elif not b_name and not c_name:
        names_match = True

    addrs_match = False
    if b_addr and c_addr:
        if b_addr == c_addr or b_addr in c_addr or c_addr in b_addr:
            addrs_match = True
    elif not b_addr and not c_addr:
        addrs_match = True

    pans_match = False
    if b_pan_code and c_pan_code and b_pan_code == c_pan_code:
        pans_match = True

    # Check if identical details:
    # 1. Both names match AND both addresses match
    # 2. Both names match AND both PANs match AND addresses match or are empty
    if names_match and addrs_match and (b_name or b_addr):
        return True
    if names_match and pans_match and addrs_match and (b_name or b_addr or b_pan_code):
        return True

    return False


def determine_inward_type(
    buyer_name: Optional[str] = None,
    buyer_address: Optional[str] = None,
    buyer_pan: Optional[str] = None,
    buyer_gst: Optional[str] = None,
    consignee_name: Optional[str] = None,
    consignee_address: Optional[str] = None,
    consignee_pan: Optional[str] = None,
    consignee_gst: Optional[str] = None,
    branch_gst_list: Optional[List[str]] = None,
    client_id: Optional[int] = None,
) -> InwardType:
    """
    Determine InwardType enum based on conditions:
    1. Identical buyer and consignee details -> InwardType.NORMAL (1)
    2. Else:
       - If branch_gst_list has <= 1 entity -> InwardType.THIRD_PARTY (2)
       - If consignee GST matches any GST in branch_gst_list -> InwardType.BRANCH_TRANSFER (3)
       - Otherwise -> InwardType.THIRD_PARTY (2)
    """
    if is_identical_buyer_consignee(
        buyer_name=buyer_name,
        buyer_address=buyer_address,
        buyer_pan=buyer_pan,
        buyer_gst=buyer_gst,
        consignee_name=consignee_name,
        consignee_address=consignee_address,
        consignee_pan=consignee_pan,
        consignee_gst=consignee_gst,
    ):
        return InwardType.NORMAL

    # Fetch branch GST list from indexer if client_id provided and branch_gst_list is None
    if branch_gst_list is None and client_id is not None:
        try:
            from app.db.indexer import get_client_index
            branch_entities = get_client_index("branch", client_id)
            branch_gst_list = [
                str(b["gst_number"]) for b in branch_entities
                if isinstance(b, dict) and b.get("gst_number")
            ]
        except Exception:
            branch_gst_list = []
    print("Branch GST List:", branch_gst_list)
    print("Consignee GST:", consignee_gst)
    # Clean and filter branch_gst_list
    normalized_branch_gsts: List[str] = []
    if branch_gst_list:
        for bgst in branch_gst_list:
            norm = _normalize_gst(bgst)
            if norm:
                normalized_branch_gsts.append(norm)

    # Rule: If list of GST numbers of branch has only one entity (or 0),
    # do not match the GST number directly mark it as third party.
    if len(normalized_branch_gsts) <= 1:
        return InwardType.THIRD_PARTY

    # Check consignee GST against branch GST numbers
    norm_c_gst = _normalize_gst(consignee_gst)
    if norm_c_gst and norm_c_gst in normalized_branch_gsts:
        return InwardType.BRANCH_TRANSFER

    return InwardType.THIRD_PARTY


def determine_inward_type_from_dict(
    data: Dict[str, Any],
    branch_gst_list: Optional[List[str]] = None,
    client_id: Optional[int] = None,
) -> InwardType:
    """
    Extract buyer and consignee details from dictionary of parsed fields
    and compute InwardType enum.
    """
    def _val(k: str) -> Optional[str]:
        v = data.get(k)
        if isinstance(v, dict):
            return str(v.get("value") or v.get("label") or v.get("raw") or "") or None
        if v is not None:
            res = str(v).strip()
            return res if res else None
        return None

    buyer_name = _val("buyer_name") or _val("buyer") or _val("buyer_details") or _val("client_master")
    buyer_address = _val("buyer_address") or _val("buyer_addr") or _val("dispatch_from_delivery_address")
    buyer_pan = _val("buyer_pan") or _val("buyer_pan_no")
    buyer_gst = _val("buyer_gst") or _val("buyer_gst_no") or _val("buyer_gstin") or _val("buyer_gst_number")

    consignee_name = _val("consignee_name") or _val("consignee") or _val("consignee_details") or _val("ship_to_name")
    consignee_address = _val("consignee_address") or _val("consignee_addr") or _val("address") or _val("delivery_address")
    consignee_pan = _val("consignee_pan") or _val("consignee_pan_no")
    consignee_gst = _val("consignee_gst") or _val("consignee_gst_no") or _val("consignee_gstin") or _val("consignee_gst_number")

    if branch_gst_list is None:
        raw_list = data.get("branch_gst_list") or data.get("branch_gsts") or data.get("client_branch_gsts")
        if isinstance(raw_list, dict):
            raw_list = raw_list.get("value")
        if isinstance(raw_list, list):
            branch_gst_list = [str(x) for x in raw_list]

    if client_id is None and "client_id" in data:
        try:
            client_id = int(data["client_id"])
        except (ValueError, TypeError):
            client_id = None

    return determine_inward_type(
        buyer_name=buyer_name,
        buyer_address=buyer_address,
        buyer_pan=buyer_pan,
        buyer_gst=buyer_gst,
        consignee_name=consignee_name,
        consignee_address=consignee_address,
        consignee_pan=consignee_pan,
        consignee_gst=consignee_gst,
        branch_gst_list=branch_gst_list,
        client_id=client_id,
    )


def normalize_inward_type_value(val: Any) -> Optional[int]:
    """
    Normalize string or int value to valid InwardType integer (1, 2, 3).
    """
    if val is None:
        return None
    if isinstance(val, int) and val in (1, 2, 3):
        return val
    val_str = str(val).strip().lower()
    if val_str in ("1", "normal"):
        return InwardType.NORMAL.value
    if val_str in ("2", "third party", "third_party"):
        return InwardType.THIRD_PARTY.value
    if val_str in ("3", "branch transfer", "branch_transfer"):
        return InwardType.BRANCH_TRANSFER.value
    return None

