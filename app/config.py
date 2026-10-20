"""
app/config.py
─────────────
Centralised settings loaded from the .env file via python-dotenv.
All other modules import from here — never from os.environ directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Application configuration resolved from environment variables.

    Every attribute has a sensible default so the service starts without
    a .env file (useful for unit-tests that mock external dependencies).
    """

    # ── Mistral AI (Step 1 OCR) ───────────────────────────────────────────────
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"

    # ── Google Gemini AI (Step 2 Text-to-JSON Parsing) ────────────────────────
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_PARSE_MODEL: str = os.getenv("GEMINI_PARSE_MODEL", "gemini-3.8-flash")





    # ── Microservice Auth ─────────────────────────────────────────────────────
    MICROSERVICE_API_KEY: str = os.getenv("MICROSERVICE_API_KEY", "")

    # ── MySQL ─────────────────────────────────────────────────────────────────
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "erp_db")
    DB_USER: str = os.getenv("DB_USER", "readonly_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_READ_TIMEOUT: int = int(os.getenv("DB_READ_TIMEOUT", "10"))

    # ── Entity Index ──────────────────────────────────────────────────────────
    ENTITY_INDEX_TTL_MINUTES: int = int(os.getenv("ENTITY_INDEX_TTL_MINUTES", "60"))

    # ── Upload Limits ─────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        """Derived limit in bytes for direct comparison."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:4200"
        ).split(",")
        if origin.strip()
    ]

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "8001"))

    # ── Confidence thresholds ─────────────────────────────────────────────────
    CONFIDENCE_HIGH: float = 0.85
    CONFIDENCE_MEDIUM_LOW: float = 0.60

    # ── Master table definitions ──────────────────────────────────────────────
    # Maps master_key → MySQL table name + list of columns to fetch.
    # Convention:
    #   • "name_cols": list of DB column names to SELECT.
    #                  The FIRST entry becomes the "name" key used for fuzzy
    #                  matching; all entries are stored in the entity dict.
    # Extend here when new entity types are added to the ERP.
    MASTER_TABLE_MAP: dict[str, dict] = {
        "supplier":         {"table": "common_companyledgermaster",         "name_cols": ["name", "gst_number"]},
        "delivery_address": {"table": "common_deliveryaddress",              "name_cols": ["delivery_name", "address1", "address2"]},
        "transport":        {"table": "common_transportmaster",              "name_cols": ["name"]},
        "agent":            {"table": "common_agentmaster",                  "name_cols": ["name"]},
        "quality":          {"table": "trading_fabricqualitymaster",         "name_cols": ["name"]},
        "design":           {"table": "trading_fabricdesignmaster",          "name_cols": ["name"]},
        "chart":            {"table": "trading_fabricchartmaster",           "name_cols": ["name"]},
        "color":            {"table": "trading_fabriccolormaster",           "name_cols": ["name"]},
        "company":          {"table": "common_companyledgermaster",         "name_cols": ["name"]},
        "branch":           {"table": "common_branchmaster",                 "name_cols": ["name", "gst_number"]},
        "unit":             {"table": "common_unitmaster",                   "name_cols": ["short_name", "name"], "is_global": True, "has_active": False},
        # ── Job Worker Master (Alias for Company Ledger) ──────────────────────────
        "job_worker":       {"table": "common_companyledgermaster",         "name_cols": ["name", "gst_number"]},
        # ── Fabric Master Aliases ─────────────────────────────────────────────────
        "fabric_quality":   {"table": "trading_fabricqualitymaster",         "name_cols": ["name"]},
        "fabric_design":    {"table": "trading_fabricdesignmaster",          "name_cols": ["name"]},
        "fabric_chart":     {"table": "trading_fabricchartmaster",           "name_cols": ["name"]},
        "fabric_color":     {"table": "trading_fabriccolormaster",           "name_cols": ["name"]},
        # ── Article Masters ───────────────────────────────────────────────────────
        "article_code":     {"table": "trading_articlecodemaster",           "name_cols": ["article_code", "description"]},
        "article":          {"table": "trading_articlecodemaster",           "name_cols": ["article_code", "description"]},
        # ── Yarn Masters ──────────────────────────────────────────────────────────
        "yarn_article":     {"table": "common_yarnarticlemaster",           "name_cols": ["name", "short_name", "auto_generated_name"]},
        "yarn_color":       {"table": "common_yarncolormaster",             "name_cols": ["name", "short_name"]},
        "yarn_brand":       {"table": "common_yarnbrandmaster",             "name_cols": ["name", "short_name"]},
        "brand":            {"table": "common_yarnbrandmaster",             "name_cols": ["name", "short_name"]},
        # ── Greige Masters ────────────────────────────────────────────────────────
        "greige_article":   {"table": "trading_tradinggreigearticlemaster", "name_cols": ["name"]},
        "greige_color":     {"table": "trading_tradinggreigecolormaster",   "name_cols": ["name"]},
    }


    # delivery_address parent column (ledger FK) used for parent-filtered match
    DELIVERY_ADDRESS_PARENT_COL: str = "company_ledger_id"

    # Cost per page for Mistral OCR (USD)
    COST_PER_PAGE_USD: float = 0.002


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a singleton Settings instance (cached after first call).

    Returns:
        Settings: Application configuration object.
    """
    return Settings()


settings = get_settings()
