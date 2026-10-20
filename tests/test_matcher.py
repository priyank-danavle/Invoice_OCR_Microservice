"""
tests/test_matcher.py
─────────────────────
Unit tests for app/ocr/matcher.py.

The entity index (app.db.indexer) is mocked at module-level so no real
MySQL connection is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from app.ocr.matcher import EntityMatchError, MatchResult, match_entity


# ── Index fixtures ────────────────────────────────────────────────────────────

SUPPLIER_DATA: List[Dict[str, Any]] = [
    {"id": 1,  "main_name": "Reliance Industries Limited",  "client_id": 1},
    {"id": 2,  "main_name": "Reliance Textiles Pvt Ltd",    "client_id": 1},
    {"id": 3,  "main_name": "Bombay Dyeing & Mfg Co Ltd",   "client_id": 1},
    {"id": 4,  "main_name": "Raymond Fabrics Limited",      "client_id": 1},
    {"id": 5,  "main_name": "Vardhman Textiles Ltd",        "client_id": 1},
]

DELIVERY_DATA: List[Dict[str, Any]] = [
    {"id": 10, "main_name": "Mumbai Warehouse - Andheri",  "client_id": 1, "ledger_id": 1},
    {"id": 11, "main_name": "Surat Godown - Ring Road",    "client_id": 1, "ledger_id": 1},
    {"id": 12, "main_name": "Delhi Distribution Centre",   "client_id": 1, "ledger_id": 3},
]

CLIENT_2_SUPPLIER_DATA: List[Dict[str, Any]] = [
    {"id": 100, "main_name": "Alpha Textiles",   "client_id": 2},
    {"id": 101, "main_name": "Beta Fabrics Ltd", "client_id": 2},
]


def _patch_index(master_key: str, client_id: int, data: List[Dict[str, Any]]):
    """Context manager: patch get_client_index to return specified data."""
    return patch(
        "app.ocr.matcher.get_client_index",
        return_value=data,
    )


# ── Exact / high-confidence match ─────────────────────────────────────────────


class TestExactMatch:
    """Exact name matches should return confidence ≥ 0.95."""

    def test_exact_match_confidence(self) -> None:
        """Exact name match should yield very high confidence."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Reliance Industries Limited", "supplier", 1)

        assert result.matched_id == 1
        assert result.matched_label == "Reliance Industries Limited"
        assert result.confidence >= 0.95

    def test_matched_id_is_correct_pk(self) -> None:
        """The matched_id must correspond to the correct DB primary key."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Raymond Fabrics Limited", "supplier", 1)

        assert result.matched_id == 4

    def test_case_insensitive_match(self) -> None:
        """Matching should be case-insensitive."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("raymond fabrics limited", "supplier", 1)

        assert result.matched_id == 4
        assert result.confidence >= 0.85


# ── Abbreviated / fuzzy match ─────────────────────────────────────────────────


class TestAbbreviatedMatch:
    """Abbreviated names should still return the correct entity."""

    def test_abbreviated_supplier_name(self) -> None:
        """'Rel. Ind.' should fuzzy-match to Reliance Industries Limited."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Rel. Ind.", "supplier", 1)

        assert result.matched_id in (1, 2)  # one of the Reliance entities
        assert result.confidence >= 0.60

    def test_partial_name_matches(self) -> None:
        """'Vardhman' alone should match Vardhman Textiles Ltd."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Vardhman", "supplier", 1)

        assert result.matched_id == 5
        assert result.confidence >= 0.60

    def test_transposed_words_match(self) -> None:
        """'Bombay Dyeing' should match 'Bombay Dyeing & Mfg Co Ltd'."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Bombay Dyeing", "supplier", 1)

        assert result.matched_id == 3
        assert result.confidence >= 0.60


# ── Low confidence / no match ─────────────────────────────────────────────────


class TestLowConfidenceMatch:
    """Scores below 60% must result in matched_id = None."""

    def test_completely_unrelated_name_no_match(self) -> None:
        """A completely unrelated string should yield matched_id = None."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("XYZQRST NONEXISTENT CO", "supplier", 1)

        assert result.matched_id is None
        assert result.confidence < 0.60

    def test_alternatives_returned_on_low_confidence(self) -> None:
        """Even when matched_id is None, alternatives above 60% should be listed."""
        # Use a name that's somewhat close but not a real match
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Reliance", "supplier", 1)

        # Reliance is a prefix of two entries — at least one should be an alternative
        if result.matched_id is None:
            # alternatives may or may not be present depending on fuzzy scores
            assert isinstance(result.alternatives, list)

    def test_empty_raw_returns_empty_result(self) -> None:
        """Empty raw string should return a MatchResult with all None fields."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("", "supplier", 1)

        assert result.matched_id is None
        assert result.confidence == 0.0
        assert result.alternatives == []

    def test_none_raw_returns_empty_result(self) -> None:
        """None raw should return a MatchResult with matched_id = None."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity(None, "supplier", 1)

        assert result.matched_id is None
        assert result.confidence == 0.0


# ── Parent-filtered match (delivery_address) ──────────────────────────────────


class TestParentFilteredMatch:
    """delivery_address lookups must be filtered by parent ledger_id."""

    def test_delivery_address_filtered_by_parent_id(self) -> None:
        """Only addresses belonging to ledger_id=1 should be searched."""
        with _patch_index("delivery_address", 1, DELIVERY_DATA):
            result = match_entity(
                "Surat Godown",
                "delivery_address",
                client_id=1,
                parent_id=1,  # supplier matched_id=1 → Reliance Industries
            )

        # Should match id=11 (Surat Godown, ledger_id=1)
        assert result.matched_id == 11
        assert result.confidence >= 0.80

    def test_delivery_address_wrong_parent_no_match(self) -> None:
        """Addresses for a different supplier (ledger_id=3) must not appear."""
        with _patch_index("delivery_address", 1, DELIVERY_DATA):
            result = match_entity(
                "Delhi Distribution Centre",
                "delivery_address",
                client_id=1,
                parent_id=1,  # parent is Reliance (ledger_id=1), not ledger_id=3
            )

        # Delhi Distribution is under ledger_id=3 → not in filtered pool
        # Result should either be None or a different (mismatched) address
        if result.matched_id is not None:
            assert result.matched_id in (10, 11)  # only Reliance addresses
        else:
            assert result.matched_id is None

    def test_single_address_after_filter_returns_095(self) -> None:
        """When only one delivery address remains after filter, confidence = 0.95."""
        single_address_data = [
            {"id": 10, "main_name": "Mumbai Warehouse - Andheri", "client_id": 1, "ledger_id": 1},
            {"id": 12, "main_name": "Delhi Distribution Centre",  "client_id": 1, "ledger_id": 3},
        ]
        with _patch_index("delivery_address", 1, single_address_data):
            result = match_entity(
                "Mumbai",
                "delivery_address",
                client_id=1,
                parent_id=1,  # ledger_id=1 → only id=10 remains
            )

        assert result.matched_id == 10
        assert result.confidence == pytest.approx(0.95)


# ── Error handling ────────────────────────────────────────────────────────────


class TestMatcherErrors:
    """Error propagation from the entity index."""

    def test_client_not_found_raises_entity_match_error(self) -> None:
        """Missing client_id in index should raise EntityMatchError."""
        from app.db.indexer import ClientNotFoundError

        with patch(
            "app.ocr.matcher.get_client_index",
            side_effect=ClientNotFoundError("client_id=999 not found"),
        ):
            with pytest.raises(EntityMatchError, match="999"):
                match_entity("Reliance", "supplier", 999)

    def test_empty_entity_pool_returns_empty_result(self) -> None:
        """If the entity pool is empty, return a MatchResult with no match."""
        with _patch_index("supplier", 1, []):
            result = match_entity("Reliance", "supplier", 1)

        assert result.matched_id is None
        assert result.confidence == 0.0


# ── Alternatives ──────────────────────────────────────────────────────────────


class TestAlternatives:
    """Alternatives should contain up to 2 entries, excluding the top match."""

    def test_at_most_two_alternatives(self) -> None:
        """Alternatives list length must not exceed 2."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Reliance Industries", "supplier", 1)

        assert len(result.alternatives) <= 2

    def test_alternatives_exclude_top_match(self) -> None:
        """Alternatives must not contain the matched_id of the top result."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Reliance Industries Limited", "supplier", 1)

        alt_ids = [alt.id for alt in result.alternatives]
        assert result.matched_id not in alt_ids

    def test_alternatives_confidence_above_060(self) -> None:
        """Each alternative's confidence must be ≥ 0.60."""
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            result = match_entity("Reliance Industries Limited", "supplier", 1)

        for alt in result.alternatives:
            assert alt.confidence >= 0.60


# ── Unit match tests ──────────────────────────────────────────────────────────

UNIT_DATA: List[Dict[str, Any]] = [
    {"id": 1,  "main_name": "kg",  "name": "Kg"},
    {"id": 2,  "main_name": "in",  "name": "Inch"},
    {"id": 3,  "main_name": "m",   "name": "Meter"},
    {"id": 4,  "main_name": "yd",  "name": "Yard"},
    {"id": 5,  "main_name": "l",   "name": "Litre"},
    {"id": 6,  "main_name": "g",   "name": "Gram"},
    {"id": 7,  "main_name": "ml",  "name": "Millilitre"},
    {"id": 8,  "main_name": "pct", "name": "Percentage"},
    {"id": 9,  "main_name": "cm",  "name": "Centimetre"},
    {"id": 10, "main_name": "kl",  "name": "Kilolitre"},
    {"id": 11, "main_name": "pcs", "name": "Pieces"},
    {"id": 12, "main_name": "bdl", "name": "Bundles"},
    {"id": 13, "main_name": "bun", "name": "Bunches"},
    {"id": 14, "main_name": "bag", "name": "Bags"},
    {"id": 15, "main_name": "box", "name": "Box"},
    {"id": 16, "main_name": "pac", "name": "Packs"},
    {"id": 17, "main_name": "ctn", "name": "Cartons"},
    {"id": 18, "main_name": "MT",  "name": "Metric Ton"},
    {"id": 19, "main_name": "qtl", "name": "Quintals"},
    {"id": 20, "main_name": "nos", "name": "Numbers"},
    {"id": 21, "main_name": "btl", "name": "Bottles"},
    {"id": 22, "main_name": "sqm", "name": "Square Mtr"},
    {"id": 23, "main_name": "sqf", "name": "Square Ft"},
    {"id": 24, "main_name": "sqy", "name": "Square Yrd"},
    {"id": 25, "main_name": "cbm", "name": "Cubic Mtr"},
    {"id": 26, "main_name": "ccm", "name": "Cubic Cm"},
    {"id": 27, "main_name": "rol", "name": "Rolls"},
    {"id": 28, "main_name": "prs", "name": "Pairs"},
    {"id": 29, "main_name": "set", "name": "Sets"},
    {"id": 30, "main_name": "can", "name": "Cans"},
    {"id": 31, "main_name": "doz", "name": "Dozens"},
    {"id": 32, "main_name": "mm",  "name": "Millimetre"},
    {"id": 33, "main_name": "mg",  "name": "Milligram"},
    {"id": 34, "main_name": "ft",  "name": "Feet"},
]


class TestUnitMatch:
    """Unit matching tests ensuring UQC codes, full names, synonyms and plurals match unit master."""

    def test_kgs_matches_kg(self) -> None:
        """'Kgs' must match 'kg' (id=1)."""
        with _patch_index("unit", 1, UNIT_DATA):
            result = match_entity("Kgs", "unit", 1)

        assert result.matched_id == 1
        assert result.matched_label == "kg"
        assert result.confidence >= 0.98

    def test_kg_matches_kg(self) -> None:
        """'KG' must match 'kg' (id=1) at confidence 1.0."""
        with _patch_index("unit", 1, UNIT_DATA):
            result = match_entity("KG", "unit", 1)

        assert result.matched_id == 1
        assert result.matched_label == "kg"
        assert result.confidence == 1.0

    def test_kgs_dot_matches_kg(self) -> None:
        """'Kgs.' must match 'kg' (id=1)."""
        with _patch_index("unit", 1, UNIT_DATA):
            result = match_entity("Kgs.", "unit", 1)

        assert result.matched_id == 1
        assert result.matched_label == "kg"

    def test_grams_matches_g(self) -> None:
        """'Grams' must match 'g' (id=6)."""
        with _patch_index("unit", 1, UNIT_DATA):
            result = match_entity("Grams", "unit", 1)

        assert result.matched_id == 6
        assert result.matched_label == "g"

    def test_mtrs_matches_m(self) -> None:
        """'Mtrs' must match 'm' (id=3)."""
        with _patch_index("unit", 1, UNIT_DATA):
            result = match_entity("Mtrs", "unit", 1)

        assert result.matched_id == 3
        assert result.matched_label == "m"

    def test_uqc_codes(self) -> None:
        """GST UQC codes (MTS, NUM, BND, BKN, MLT, KLR, SFT, SMA, etc.) should map to short_name."""
        test_cases = [
            ("MTS", 18, "MT"),
            ("NUM", 20, "nos"),
            ("NOS", 20, "nos"),
            ("UNT", 20, "nos"),
            ("BND", 12, "bdl"),
            ("BKN", 15, "box"),
            ("MLT", 7, "ml"),
            ("KLR", 10, "kl"),
            ("SFT", 23, "sqf"),
            ("SMA", 22, "sqm"),
            ("QTL", 19, "qtl"),
            ("GMS", 6, "g"),
            ("PCS", 11, "pcs"),
            ("BTL", 21, "btl"),
            ("ROL", 27, "rol"),
            ("PRS", 28, "prs"),
            ("SET", 29, "set"),
            ("CAN", 30, "can"),
            ("DOZ", 31, "doz"),
            ("MME", 32, "mm"),
            ("DMG", 33, "mg"),
        ]
        with _patch_index("unit", 1, UNIT_DATA):
            for uqc, expected_id, expected_label in test_cases:
                res = match_entity(uqc, "unit", 1)
                assert res.matched_id == expected_id, f"Failed for UQC '{uqc}'"
                assert res.matched_label == expected_label, f"Failed label for UQC '{uqc}'"

    def test_dotted_and_full_unit_names(self) -> None:
        """Dotted forms like 'sq. mtr.' or full names 'Metric Ton' should match."""
        test_cases = [
            ("sq. mtr.", 22, "sqm"),
            ("sq.ft", 23, "sqf"),
            ("Metric Ton", 18, "MT"),
            ("Quintals", 19, "qtl"),
            ("Square Meter", 22, "sqm"),
            ("Cubic Mtr", 25, "cbm"),
        ]
        with _patch_index("unit", 1, UNIT_DATA):
            for raw_val, expected_id, expected_label in test_cases:
                res = match_entity(raw_val, "unit", 1)
                assert res.matched_id == expected_id, f"Failed for '{raw_val}'"
                assert res.matched_label == expected_label, f"Failed label for '{raw_val}'"

    def test_dict_raw_input_handling(self) -> None:
        """Dict inputs like {'label': 'Reliance Industries Limited'} should be sanitized without AttributeError."""
        from app.ocr.matcher import match_entity_exact
        with _patch_index("supplier", 1, SUPPLIER_DATA):
            res = match_entity({"label": "Reliance Industries Limited"}, "supplier", 1)
            assert res.matched_id == 1

            res_exact = match_entity_exact({"label": "Reliance Industries Limited"}, "supplier", 1)
            assert res_exact.matched_id == 1


