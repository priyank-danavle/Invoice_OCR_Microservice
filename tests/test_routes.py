import json
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app

client = TestClient(app)
API_KEY = settings.MICROSERVICE_API_KEY

def test_parse_route_variations_without_auth():
    endpoints = [
        "/api/v1/invoice/parse",
        "/api/v1/invoice/parse/",
        "/ocr-api/api/v1/invoice/parse",
        "/ocr-api/api/v1/invoice/parse/",
    ]
    for ep in endpoints:
        res = client.post(ep)
        assert res.status_code == 401, f"POST {ep} expected 401, got {res.status_code}"

def test_extract_route_variations_without_auth():
    endpoints = [
        "/api/v1/invoice/extract",
        "/api/v1/invoice/extract/",
        "/ocr-api/api/v1/invoice/extract",
        "/ocr-api/api/v1/invoice/extract/",
    ]
    for ep in endpoints:
        res = client.post(ep)
        assert res.status_code == 401, f"POST {ep} expected 401, got {res.status_code}"

def test_process_route_variations_without_auth():
    endpoints = [
        "/api/v1/invoice/process",
        "/api/v1/invoice/process/",
        "/ocr-api/api/v1/invoice/process",
        "/ocr-api/api/v1/invoice/process/",
    ]
    for ep in endpoints:
        res = client.post(ep)
        assert res.status_code == 401, f"POST {ep} expected 401, got {res.status_code}"

def test_health_route_variations():
    endpoints = [
        "/health",
        "/health/",
        "/ocr-api/health",
        "/ocr-api/health/",
    ]
    for ep in endpoints:
        res = client.get(ep)
        assert res.status_code == 200, f"GET {ep} expected 200, got {res.status_code}"


def test_direct_field_result_accepts_numeric_values():
    from app.models.response import DirectFieldResult
    res_float = DirectFieldResult(value=1437.14, confidence=1.0)
    assert res_float.value == 1437.14
    assert res_float.type == "direct"

    res_int = DirectFieldResult(value=100, confidence=0.9)
    assert res_int.value == 100

    res_str = DirectFieldResult(value="1437.14", confidence=1.0)
    assert res_str.value == "1437.14"


def test_process_inward_type_normal():
    headers = {"X-API-Key": API_KEY}

    schema = {
        "supplier_name": {"type": "direct"},
        "inward_type": {"type": "direct"}
    }
    extracted = {
        "buyer_name": "Ginza Industries Ltd.",
        "buyer_address": "Plot A-2, Surat",
        "consignee_name": "Ginza Industries Ltd.",
        "consignee_address": "Plot A-2, Surat",
        "supplier_name": "Ginza Industries Ltd."
    }

    res = client.post(
        "/api/v1/invoice/process",
        headers=headers,
        data={
            "client_id": "1",
            "output_schema": json.dumps(schema),
            "raw_text": json.dumps(extracted)
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["basic_details"]["inward_type"] == 1


def test_process_inward_type_branch_transfer():
    headers = {"X-API-Key": API_KEY}

    schema = {
        "inward_type": {"type": "direct"}
    }
    extracted = {
        "buyer_name": "Ace Tex Surat",
        "buyer_address": "Surat, Gujarat",
        "buyer_gst": "24ABOFA5531B1ZU",
        "consignee_name": "Ace Tex Mumbai",
        "consignee_address": "Mumbai, Maharashtra",
        "consignee_gst": "27ABOFA5531B1Z5",
        "branch_gst_list": ["24ABOFA5531B1ZU", "27ABOFA5531B1Z5"],
    }

    res = client.post(
        "/api/v1/invoice/process",
        headers=headers,
        data={
            "client_id": "1",
            "output_schema": json.dumps(schema),
            "raw_text": json.dumps(extracted)
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["basic_details"]["inward_type"] == 3


def test_process_inward_type_third_party():
    headers = {"X-API-Key": API_KEY}

    schema = {
        "inward_type": {"type": "direct"}
    }
    extracted = {
        "buyer_name": "Ace Tex",
        "buyer_address": "Surat",
        "buyer_gst": "24ABOFA5531B1ZU",
        "consignee_name": "Raj",
        "consignee_address": "Port Blair",
        "consignee_gst": ""
    }

    res = client.post(
        "/api/v1/invoice/process",
        headers=headers,
        data={
            "client_id": "1",
            "output_schema": json.dumps(schema),
            "raw_text": json.dumps(extracted)
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["basic_details"]["inward_type"] == 2


