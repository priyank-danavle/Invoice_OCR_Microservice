"""
tests/test_inward_classifier.py
────────────────────────────────
Unit tests for InwardType enum and inward classification logic.
"""

from app.models.enums import InwardType
from app.ocr.inward_classifier import (
    determine_inward_type,
    determine_inward_type_from_dict,
    extract_pan,
    is_identical_buyer_consignee,
)


def test_inward_type_enum_values():
    """Verify that InwardType enum values match the integer specification (1, 2, 3)."""
    assert InwardType.NORMAL.value == 1
    assert InwardType.THIRD_PARTY.value == 2
    assert InwardType.BRANCH_TRANSFER.value == 3


def test_extract_pan():
    """Test PAN extraction from direct PAN string and from GSTIN string."""
    # Direct PAN
    assert extract_pan("AABCG0675P") == "AABCG0675P"
    assert extract_pan("ABOFA5531B") == "ABOFA5531B"
    assert extract_pan("PAN: ABOFA5531B") == "ABOFA5531B"

    # From GSTIN (15 characters: 2 digits + 10 char PAN + 3 chars)
    assert extract_pan("24AABCG0675P1ZD") == "AABCG0675P"
    assert extract_pan("24ABOFA5531B1ZU") == "ABOFA5531B"
    assert extract_pan("27AACFW1070C1ZQ") == "AACFW1070C"
    assert extract_pan("GST No: 24ABOFA5531B1ZU") == "ABOFA5531B"

    # Invalid / empty
    assert extract_pan("") is None
    assert extract_pan(None) is None
    assert extract_pan("INVALID_PAN") is None


def test_pdf1_identical_buyer_consignee_normal():
    """
    Test PDF 1 scenario (Ginza Industries Ltd.):
    Buyer and Consignee details are identical -> Should return Normal (1).
    """
    res = determine_inward_type(
        buyer_name="Ginza Industries Ltd. (Ckn) / Circular Knitting",
        buyer_address="Plot No A-2/10, Roadno 10, Hoziwala Industrial Estate, Sachin, Surat - 394230",
        buyer_pan="AABCG0675P",
        buyer_gst="24AABCG0675P1ZD",
        consignee_name="Ginza Industries Ltd. (Ckn)",
        consignee_address="Plot No A-2/10, Roadno 10, Hoziwala Industrial Estate, Sachin, Surat - 394230",
        consignee_pan="AABCG0675P",
        consignee_gst="24AABCG0675P1ZD",
    )
    assert res == InwardType.NORMAL
    assert res.value == 1


def test_pdf2_identical_buyer_consignee_normal():
    """
    Test PDF 2 scenario (Ace Tex to Ace Tex):
    Buyer and Consignee details are identical -> Should return Normal (1).
    """
    res = determine_inward_type(
        buyer_name="Ace Tex",
        buyer_address="Surat, Surat, Surat - 395009",
        buyer_pan="ABOFA5531B",
        buyer_gst="24ABOFA5531B1ZU",
        consignee_name="Ace Tex",
        consignee_address="Surat, Surat, Surat - 395009",
        consignee_pan="ABOFA5531B",
        consignee_gst="24ABOFA5531B1ZU",
    )
    assert res == InwardType.NORMAL
    assert res.value == 1


def test_pdf3_different_buyer_consignee_third_party():
    """
    Test PDF 3 scenario (Ace Tex to Raj):
    Buyer and Consignee details differ, different/missing PAN -> Should return third party (2).
    """
    res = determine_inward_type(
        buyer_name="Ace Tex",
        buyer_address="Surat, Surat, Surat - 395009",
        buyer_pan="ABOFA5531B",
        buyer_gst="24ABOFA5531B1ZU",
        consignee_name="Raj",
        consignee_address="vesu, vesu, Port Blair - 009988",
        consignee_pan="",
        consignee_gst="",
    )
    assert res == InwardType.THIRD_PARTY
    assert res.value == 2


def test_branch_transfer_multiple_branches_match():
    """
    Test Branch Transfer scenario:
    Buyer and Consignee details differ, branch_gst_list has > 1 entities,
    and consignee GST matches one of the branch GST numbers -> Should return BRANCH_TRANSFER (3).
    """
    res = determine_inward_type(
        buyer_name="Ace Tex Surat",
        buyer_address="Plot 12, Surat, Gujarat",
        buyer_gst="24ABOFA5531B1ZU",
        consignee_name="Ace Tex Mumbai Branch",
        consignee_address="Building 4, Mumbai, Maharashtra",
        consignee_gst="27ABOFA5531B1Z5",
        branch_gst_list=["24ABOFA5531B1ZU", "27ABOFA5531B1Z5"],
    )
    assert res == InwardType.BRANCH_TRANSFER
    assert res.value == 3


def test_single_branch_entity_returns_third_party():
    """
    Test Single Branch Entity scenario:
    If list of GST numbers of branch has ONLY 1 entity, even if consignee GST matches that entity,
    do NOT match the GST number directly mark it as THIRD_PARTY (2).
    """
    res = determine_inward_type(
        buyer_name="Ace Tex Surat",
        buyer_address="Plot 12, Surat, Gujarat",
        buyer_gst="24ABOFA5531B1ZU",
        consignee_name="Ace Tex Mumbai Branch",
        consignee_address="Building 4, Mumbai, Maharashtra",
        consignee_gst="27ABOFA5531B1Z5",
        branch_gst_list=["27ABOFA5531B1Z5"],
    )
    assert res == InwardType.THIRD_PARTY
    assert res.value == 2


def test_branch_gst_no_match_returns_third_party():
    """
    Test scenario where branch_gst_list has > 1 entities, but consignee GST does not match any -> THIRD_PARTY (2).
    """
    res = determine_inward_type(
        buyer_name="Ace Tex Surat",
        buyer_address="Plot 12, Surat, Gujarat",
        buyer_gst="24ABOFA5531B1ZU",
        consignee_name="Third Party Customer",
        consignee_address="Building 9, Delhi",
        consignee_gst="07ABCDE1234F1Z9",
        branch_gst_list=["24ABOFA5531B1ZU", "27ABOFA5531B1Z5"],
    )
    assert res == InwardType.THIRD_PARTY
    assert res.value == 2


def test_determine_inward_type_from_dict():
    """Test determine_inward_type_from_dict with dictionary input containing parsed fields and branch_gst_list."""
    data = {
        "buyer_name": {"value": "Ace Tex"},
        "buyer_address": "Surat, Surat - 395009",
        "buyer_gst": "24ABOFA5531B1ZU",
        "consignee_name": "Ace Tex Branch",
        "consignee_address": "Vapi, Gujarat",
        "consignee_gst": "24ABOFA5531B1Z9",
        "branch_gst_list": ["24ABOFA5531B1ZU", "24ABOFA5531B1Z9"],
    }
    assert determine_inward_type_from_dict(data) == InwardType.BRANCH_TRANSFER
    assert determine_inward_type_from_dict(data).value == 3

