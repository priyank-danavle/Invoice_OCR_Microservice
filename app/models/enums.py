"""
app/models/enums.py
───────────────────
Enum definitions for OCR microservice.
"""

from enum import IntEnum


class InwardType(IntEnum):
    """
    Enum representing the type of inward document:
    - NORMAL (1): Buyer and consignee details are identical.
    - THIRD_PARTY (2): Buyer and consignee details differ and PAN numbers do not match.
    - BRANCH_TRANSFER (3): Buyer and consignee details differ, but both share the same PAN number.
    """
    NORMAL = 1
    THIRD_PARTY = 2
    BRANCH_TRANSFER = 3
