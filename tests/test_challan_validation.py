"""
tests/test_challan_validation.py
──────────────────────────────────
Unit and integration tests for single delivery challan upload validation.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app.main import app
from app.ocr.challan_validator import (
    MultipleChallansDetectedError,
    check_and_raise_single_challan,
    validate_single_challan,
)

client = TestClient(app)

API_KEY = "fc8fe7865dfab583b09b5e3c25f06fac3bcb45ca1958f8a912bf321837455d19"


def test_single_page_single_challan_passes() -> None:
    text = """
    DELIVERY CHALLAN
    Challan No: DC-1001
    Date: 15/08/2026
    Buyer: Reliance Industries Ltd, Mumbai
    Consignee: Reliance Industries Ltd, Surat Factory
    Item: Cotton Fabric 100m
    """
    is_valid, err_msg = validate_single_challan(text)
    assert is_valid is True
    assert err_msg is None


def test_multi_page_single_challan_passes() -> None:
    text = """
    === File: invoice_page1.pdf ===
    DELIVERY CHALLAN
    Challan No: DC-1001
    Date: 15/08/2026
    Buyer: Reliance Industries Ltd
    Consignee: Reliance Industries Ltd
    Page 1 of 2
    
    === File: invoice_page2.pdf ===
    DELIVERY CHALLAN (Continuation)
    Challan No: DC-1001
    Buyer: Reliance Industries Ltd
    Consignee: Reliance Industries Ltd
    Page 2 of 2
    Item: Velvet Fabric 50m
    """
    is_valid, err_msg = validate_single_challan(text)
    assert is_valid is True
    assert err_msg is None


def test_multiple_challans_with_different_buyers_fails() -> None:
    text = """
    === File: file1.pdf ===
    DELIVERY CHALLAN
    Challan No: DC-1001
    Buyer: Reliance Industries Ltd
    Consignee: Reliance Surat Branch
    
    === File: file2.pdf ===
    DELIVERY CHALLAN
    Challan No: DC-2002
    Buyer: Aditya Birla Fashion Ltd
    Consignee: Aditya Birla Warehouse
    """
    is_valid, err_msg = validate_single_challan(text)
    assert is_valid is False
    assert err_msg is not None
    assert "Multiple delivery challans detected" in err_msg
    assert "Buyer" in err_msg

    with pytest.raises(MultipleChallansDetectedError):
        check_and_raise_single_challan(text)


def test_multiple_challans_with_different_consignees_fails() -> None:
    text = """
    === File: file1.pdf ===
    DELIVERY CHALLAN
    Challan No: DC-1001
    Buyer: Arvind Mills Ltd
    Consignee: Arvind Mills Mill No 1
    
    === File: file2.pdf ===
    DELIVERY CHALLAN
    Challan No: DC-2002
    Buyer: Arvind Mills Ltd
    Consignee: Bombay Dyeing Factory 5
    """
    is_valid, err_msg = validate_single_challan(text)
    assert is_valid is False
    assert err_msg is not None
    assert "Consignee" in err_msg or "Challan Nos" in err_msg


def test_multiple_distinct_challan_numbers_fails() -> None:
    text = """
    DELIVERY CHALLAN 1
    Challan No: DC-8801
    Buyer: Raymond Ltd
    Consignee: Raymond Factory
    
    DELIVERY CHALLAN 2
    Challan No: DC-9909
    Buyer: Raymond Ltd
    Consignee: Raymond Factory
    """
    is_valid, err_msg = validate_single_challan(text)
    assert is_valid is False
    assert err_msg is not None
    assert "DC-8801" in err_msg or "DC-9909" in err_msg


def test_api_endpoint_returns_400_for_multiple_challans() -> None:
    multi_challan_text = """
    === File: c1.pdf ===
    Challan No: DC-101
    Buyer: Alpha Textiles Ltd
    Consignee: Alpha Surat Branch
    
    === File: c2.pdf ===
    Challan No: DC-909
    Buyer: Beta Fabrics Pvt Ltd
    Consignee: Beta Ahmedabad Depot
    """
    with patch("app.main.extract", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = (multi_challan_text, 2)
        res = client.post(
            "/api/v1/invoice/extract",
            headers={"X-API-Key": API_KEY},
            data={"client_id": "1"},
            files=[("file", ("c1.pdf", b"%PDF-1.4 dummy", "application/pdf"))],
        )

    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "multiple_challans_detected"
    assert "Multiple delivery challans detected" in body["message"]
