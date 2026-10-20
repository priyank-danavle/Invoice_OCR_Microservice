"""
app/ocr/matcher.py
──────────────────
Generic fuzzy entity matcher backed by the in-memory entity index.

match_entity() is the single public function consumed by the route handler.
It handles parent-filtered lookups (e.g. delivery_address filtered by
ledger_id == supplier matched_id) transparently.

Confidence scale mapping from rapidfuzz scores (0–100) to normalised
floats (0.0–1.0):
    ≥ 85  → HIGH   (≥ 0.85)
    60–84 → MEDIUM (0.60–0.84)
    < 60  → LOW    (< 0.60) → matched_id = None, alternatives only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz, process as fuzz_process

from app.config import settings
from app.db.indexer import ClientNotFoundError, get_client_index

import os as _os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# ── Dedicated file logger for color-matching debug output ─────────────────────
# Writes to /app/logs/color_debug.log (host path: ./logs/color_debug.log via
# the Docker volume mount).  Kept separate from the main logger so the file
# only contains the [COLOR_DEBUG] lines — easy to tail/read without noise.
_LOG_DIR  = _os.getenv("LOG_DIR", "/app/logs")
try:
    _os.makedirs(_LOG_DIR, exist_ok=True)
except PermissionError:
    _LOG_DIR = "./logs"
    _os.makedirs(_LOG_DIR, exist_ok=True)

_LOG_FILE = _os.path.join(_LOG_DIR, "color_debug.log")

_color_debug_logger = logging.getLogger("color_debug")
_color_debug_logger.setLevel(logging.DEBUG)
_color_debug_logger.propagate = False   # don't send to root/console

if not _color_debug_logger.handlers:
    try:
        _fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s │ %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        _color_debug_logger.addHandler(_fh)
    except Exception:
        _color_debug_logger.addHandler(logging.NullHandler())


# ── Custom exceptions ─────────────────────────────────────────────────────────

class EntityMatchError(Exception):
    """Raised when the entity index cannot service a match request."""


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class AlternativeResult:
    """A runner-up candidate from fuzzy matching."""
    id: int
    label: str
    confidence: float


@dataclass
class MatchResult:
    """
    Full result of a single entity match attempt.

    Attributes:
        raw: The raw string extracted from OCR (passed through unchanged).
        matched_id: PK of the best match, or None if confidence < LOW threshold.
        matched_label: Human-readable label of the best match.
        confidence: Normalised score in [0.0, 1.0].
        alternatives: Up to 2 runner-up matches above MEDIUM threshold.
        master: The master_key used for this match.
        extra_data: Any additional columns fetched from the master table.
    """
    raw: Optional[str]
    matched_id: Optional[int]
    matched_label: Optional[str]
    confidence: float
    alternatives: List[AlternativeResult] = field(default_factory=list)
    master: str = ""
    extra_data: Dict[str, Any] = field(default_factory=dict)


UNIT_SYNONYMS: Dict[str, str] = {
    # 1: Kg (kg)
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",

    # 2: Inch (in)
    "in": "in", "inc": "in", "inch": "in", "inches": "in",

    # 3: Meter (m)
    "m": "m", "mtr": "m", "mtrs": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",

    # 4: Yard (yd)
    "yd": "yd", "yds": "yd", "gyd": "yd", "yard": "yd", "yards": "yd",

    # 5: Litre (l)
    "l": "l", "ltr": "l", "ltrs": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",

    # 6: Gram (g)
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",

    # 7: Millilitre (ml)
    "ml": "ml", "mlt": "ml", "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml",

    # 8: Percentage (pct)
    "pct": "pct", "%": "pct", "percent": "pct", "percentage": "pct",

    # 9: Centimetre (cm)
    "cm": "cm", "cmt": "cm", "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm",

    # 10: Kilolitre (kl)
    "kl": "kl", "klr": "kl", "kilolitre": "kl", "kilolitres": "kl", "kiloliter": "kl", "kiloliters": "kl",

    # 11: Pieces (pcs)
    "pcs": "pcs", "pc": "pcs", "piece": "pcs", "pieces": "pcs",

    # 12: Bundles (bdl)
    "bdl": "bdl", "bnd": "bdl", "bundle": "bdl", "bundles": "bdl", "bkl": "bdl",

    # 13: Bunches (bun)
    "bun": "bun", "bunch": "bun", "bunches": "bun",

    # 14: Bags (bag)
    "bag": "bag", "bags": "bag",

    # 15: Box (box)
    "box": "box", "boxes": "box", "bkn": "box",

    # 16: Packs (pac)
    "pac": "pac", "pak": "pac", "pack": "pac", "packs": "pac", "pkt": "pac", "pkts": "pac", "packet": "pac", "packets": "pac",

    # 17: Cartons (ctn)
    "ctn": "ctn", "ct": "ctn", "carton": "ctn", "cartons": "ctn",

    # 18: Metric Ton (MT -> mt)
    "mt": "mt", "mts": "mt", "ton": "mt", "tons": "mt", "tonne": "mt", "tonnes": "mt", "metric ton": "mt", "metric tons": "mt", "metric tonne": "mt", "metric tonnes": "mt",

    # 19: Quintals (qtl)
    "qtl": "qtl", "quintal": "qtl", "quintals": "qtl",

    # 20: Numbers (nos)
    "nos": "nos", "no": "nos", "num": "nos", "number": "nos", "numbers": "nos", "unt": "nos", "unit": "nos", "units": "nos",

    # 21: Bottles (btl)
    "btl": "btl", "bottle": "btl", "bottles": "btl",

    # 22: Square Mtr (sqm)
    "sqm": "sqm", "sma": "sqm", "sq.m": "sqm", "sq m": "sqm", "sq mtr": "sqm", "sq.mtr": "sqm", "sq mtrs": "sqm", "square meter": "sqm", "square meters": "sqm", "square metre": "sqm", "square metres": "sqm", "square mtr": "sqm",

    # 23: Square Ft (sqf)
    "sqf": "sqf", "sft": "sqf", "sq.ft": "sqf", "sq ft": "sqf", "square foot": "sqf", "square feet": "sqf", "square ft": "sqf",

    # 24: Square Yrd (sqy)
    "sqy": "sqy", "sq.yd": "sqy", "sq yd": "sqy", "square yard": "sqy", "square yards": "sqy", "square yrd": "sqy",

    # 25: Cubic Mtr (cbm)
    "cbm": "cbm", "cb.m": "cbm", "cb m": "cbm", "cubic meter": "cbm", "cubic meters": "cbm", "cubic metre": "cbm", "cubic metres": "cbm", "cubic mtr": "cbm",

    # 26: Cubic Cm (ccm)
    "ccm": "ccm", "cb.cm": "ccm", "cb cm": "ccm", "cubic centimeter": "ccm", "cubic centimeters": "ccm", "cubic centimetre": "ccm", "cubic centimetres": "ccm", "cubic cm": "ccm",

    # 27: Rolls (rol)
    "rol": "rol", "rl": "rol", "rls": "rol", "roll": "rol", "rolls": "rol",

    # 28: Pairs (prs)
    "prs": "prs", "pr": "prs", "pair": "prs", "pairs": "prs",

    # 29: Sets (set)
    "set": "set", "sets": "set",

    # 30: Cans (can)
    "can": "can", "cans": "can",

    # 31: Dozens (doz)
    "doz": "doz", "dz": "doz", "dozen": "doz", "dozens": "doz", "grs": "doz",

    # 32: Millimetre (mm)
    "mm": "mm", "mme": "mm", "millimetre": "mm", "millimetres": "mm", "millimeter": "mm", "millimeters": "mm",

    # 33: Milligram (mg)
    "mg": "mg", "dmg": "mg", "milligram": "mg", "milligrams": "mg",

    # 34: Feet (ft)
    "ft": "ft", "foot": "ft", "feet": "ft",
}


def _get_canonical_unit(query_clean: str) -> Optional[str]:
    """Look up canonical unit symbol from UNIT_SYNONYMS map."""
    if query_clean in UNIT_SYNONYMS:
        return UNIT_SYNONYMS[query_clean]
    clean_nodot = query_clean.replace(".", "").strip()
    if clean_nodot in UNIT_SYNONYMS:
        return UNIT_SYNONYMS[clean_nodot]
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_entity_label(entity: Dict[str, Any]) -> str:
    """Safely return entity label from main_name or name key."""
    return str(entity.get("main_name") or entity.get("name") or "")


def _score_to_confidence(score: float) -> float:
    """
    Normalise a rapidfuzz score (0–100) to [0.0, 1.0].

    Args:
        score: rapidfuzz score in [0, 100].

    Returns:
        float: Confidence in [0.0, 1.0].
    """
    return round(score / 100.0, 4)


def _filter_by_parent(
    entities: List[Dict[str, Any]],
    parent_id: int,
    parent_col: str,
) -> List[Dict[str, Any]]:
    """
    Filter an entity list to only those whose parent_col equals parent_id.

    Used specifically for delivery_address where ledger_id == supplier matched_id.

    Args:
        entities: Full entity list for a client.
        parent_id: Matched PK of the parent entity.
        parent_col: Column name to filter on (e.g. "ledger_id" or "company_ledger_id").

    Returns:
        List[Dict]: Subset of entities matching the parent constraint.
    """
    return [
        e for e in entities
        if e.get(parent_col) == parent_id or e.get("ledger_id") == parent_id
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def match_entity(
    raw: Optional[str],
    master_key: str,
    client_id: int,
    parent_id: Optional[int] = None,
) -> MatchResult:
    """
    Fuzzy-match *raw* against the entity index for the given master_key and client.

    Algorithm:
        1. Retrieve index slice for master_key + client_id.
        2. If parent_id is given, filter to entities where ledger_id == parent_id.
        3. If only 1 entity remains after parent filter, return it at confidence 0.95.
        4. Check unit synonyms and exact / plural matches.
        5. Run rapidfuzz scoring against all candidate names of each entity.
        6. Top score ≥ 85 → HIGH; 60–84 → MEDIUM; < 60 → LOW (matched_id = None).
        7. Collect up to 2 alternatives with score ≥ 60, excluding the top match.

    Args:
        raw: Raw entity name string extracted from OCR.
             If None or empty, returns an empty MatchResult immediately.
        master_key: Entity type key (e.g. "supplier", "transport").
        client_id: ERP client ID — scopes the lookup.
        parent_id: Optional PK of a parent entity for dependent lookups.

    Returns:
        MatchResult: Full match result including alternatives.

    Raises:
        EntityMatchError: If the entity index is unavailable for the given key/client.
    """
    if isinstance(raw, dict):
        raw = raw.get("label") or raw.get("value") or raw.get("name") or (str(raw.get("id")) if raw.get("id") is not None else None)
    elif raw is not None and not isinstance(raw, str):
        raw = str(raw)

    empty_result = MatchResult(
        raw=raw,
        matched_id=None,
        matched_label=None,
        confidence=0.0,
        alternatives=[],
        master=master_key,
        extra_data={},
    )

    if not raw or not raw.strip():
        logger.debug("match_entity: raw is empty for master='%s', returning empty result.", master_key)
        return empty_result

    try:
        entities = get_client_index(master_key, client_id)
    except ClientNotFoundError as exc:
        raise EntityMatchError(str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise EntityMatchError(
            f"Failed to access entity index for master='{master_key}', "
            f"client_id={client_id}: {exc}"
        ) from exc

    # Apply parent filter if supplied
    search_pool = entities
    if parent_id is not None:
        parent_col = settings.DELIVERY_ADDRESS_PARENT_COL
        search_pool = _filter_by_parent(entities, parent_id, parent_col)
        logger.debug(
            "Parent filter applied: master='%s', parent_id=%d → %d/%d entities remain.",
            master_key, parent_id, len(search_pool), len(entities),
        )

    if not search_pool:
        logger.debug(
            "No entities to match against for master='%s', client_id=%d (after filter).",
            master_key, client_id,
        )
        return empty_result

    # Single-entity shortcut (e.g. only one delivery address for this supplier)
    if len(search_pool) == 1:
        sole = search_pool[0]
        logger.debug("Single entity after filter — returning at confidence 0.95.")
        return MatchResult(
            raw=raw,
            matched_id=int(sole["id"]),
            matched_label=_get_entity_label(sole),
            confidence=0.95,
            alternatives=[],
            master=master_key,
            extra_data={k: v for k, v in sole.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")},
        )

    # Build name → entity mapping for rapidfuzz (including secondary name columns if present)
    name_to_entity: Dict[str, Dict[str, Any]] = {}
    for entity in search_pool:
        main = str(entity.get("main_name") or entity.get("name") or "")
        name_to_entity[main] = entity
        for k, v in entity.items():
            if k not in ("id", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id") and isinstance(v, str) and v.strip():
                if v not in name_to_entity:
                    name_to_entity[v] = entity

    choices = list(name_to_entity.keys())
    query = raw.strip()
    query_clean = query.rstrip(".").strip().lower()

    # ── DEBUG: dump all candidates in the search pool ─────────────────────────
    _color_debug_logger.debug(
        "[COLOR_DEBUG] match_entity master='%s' client=%d raw='%s' — "
        "search pool has %d candidates: %s",
        master_key, client_id, query, len(choices), choices,
    )

    # ── Unit Master Synonym Lookup ─────────────────────────────────────────────
    if master_key == "unit":
        canonical_unit = _get_canonical_unit(query_clean)
        if canonical_unit:
            for candidate_name, entity in name_to_entity.items():
                if candidate_name.lower() == canonical_unit:
                    _color_debug_logger.debug(
                        "[COLOR_DEBUG] UNIT SYNONYM MATCH for query='%s' → '%s' (confidence=1.0)",
                        query, _get_entity_label(entity),
                    )
                    logger.debug(
                        "match_entity: unit synonym match master='%s' client=%d raw='%s' → '%s'",
                        master_key, client_id, query, _get_entity_label(entity),
                    )
                    return MatchResult(
                        raw=raw,
                        matched_id=int(entity["id"]),
                        matched_label=_get_entity_label(entity),
                        confidence=1.0,
                        alternatives=[],
                        master=master_key,
                        extra_data={k: v for k, v in entity.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")},
                    )

    # ── Exact & Plural match shortcut (case-insensitive) ───────────────────────
    for candidate_name, entity in name_to_entity.items():
        cand_lower = candidate_name.lower()
        if cand_lower == query_clean:
            _color_debug_logger.debug(
                "[COLOR_DEBUG] EXACT MATCH for query='%s' → '%s' (confidence=1.0)",
                query, candidate_name,
            )
            logger.debug(
                "match_entity: exact match master='%s' client=%d raw='%s' → '%s'",
                master_key, client_id, query, candidate_name,
            )
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=1.0,
                alternatives=[],
                master=master_key,
                extra_data={k: v for k, v in entity.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")},
            )
        if query_clean.endswith("s") and query_clean[:-1] == cand_lower:
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=0.98,
                alternatives=[],
                master=master_key,
                extra_data={k: v for k, v in entity.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")},
            )
        if query_clean.endswith("es") and query_clean[:-2] == cand_lower:
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=0.98,
                alternatives=[],
                master=master_key,
                extra_data={k: v for k, v in entity.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")},
            )

    # Composite scorer: take the best of WRatio, partial_ratio, token_set_ratio,
    # and an abbreviation-prefix scorer that handles inputs like "Rel. Ind."
    import re as _re

    def _abbreviation_score(q: str, choice: str) -> float:
        tokens = q.split()
        dot_tokens = [t.rstrip(".").lower() for t in tokens if t.endswith(".") and len(t) > 1]
        if not dot_tokens:
            return 0.0

        choice_words = [w.lower() for w in _re.split(r"[\s\-]+", choice) if w]
        matched = sum(
            1 for tok in dot_tokens
            if any(word.startswith(tok) for word in choice_words)
        )
        if matched == len(dot_tokens):
            return 75.0
        return (matched / len(dot_tokens)) * 59.0

    def _composite_score_with_debug(choice: str) -> float:
        choice_lower = choice.lower()
        s_wratio       = fuzz.WRatio(query_clean, choice_lower)
        s_token_set    = fuzz.token_set_ratio(query_clean, choice_lower)
        s_abbreviation = _abbreviation_score(query, choice)

        if len(choice_lower) <= 3:
            if choice_lower in query_clean.split():
                s_partial = fuzz.partial_ratio(query_clean, choice_lower)
            else:
                s_wratio  = fuzz.ratio(query_clean, choice_lower)
                s_partial = 0.0
        else:
            s_partial = fuzz.partial_ratio(query_clean, choice_lower)

        best = max(s_wratio, s_partial, s_token_set, s_abbreviation)
        _color_debug_logger.debug(
            "[COLOR_DEBUG]   candidate='%s'  WRatio=%.1f  partial_ratio=%.1f  "
            "token_set_ratio=%.1f  abbrev=%.1f  → composite=%.1f",
            choice, s_wratio, s_partial, s_token_set, s_abbreviation, best,
        )
        return best

    def _composite_score(choice: str) -> float:
        choice_lower = choice.lower()
        s_wratio       = fuzz.WRatio(query_clean, choice_lower)
        s_token_set    = fuzz.token_set_ratio(query_clean, choice_lower)
        s_abbreviation = _abbreviation_score(query, choice)

        if len(choice_lower) <= 3:
            if choice_lower in query_clean.split():
                s_partial = fuzz.partial_ratio(query_clean, choice_lower)
            else:
                s_wratio  = fuzz.ratio(query_clean, choice_lower)
                s_partial = 0.0
        else:
            s_partial = fuzz.partial_ratio(query_clean, choice_lower)

        return max(s_wratio, s_partial, s_token_set, s_abbreviation)

    # Use verbose scorer so every candidate gets logged to file, then sort
    _color_debug_logger.debug(
        "[COLOR_DEBUG] Scoring all %d candidates for query='%s' (master='%s'):",
        len(choices), query, master_key,
    )
    scored = [(name, _composite_score_with_debug(name)) for name in choices]
    # Primary sort: score descending.
    # Tiebreaker: shorter name wins — 'Green' (5) beats 'Green-2' (7) when both
    # hit 100.0 via partial_ratio (query is a perfect substring of the longer name).
    scored.sort(key=lambda x: (-x[1], len(x[0])))

    # ── DEBUG: top-5 after sorting ────────────────────────────────────────────
    _color_debug_logger.debug(
        "[COLOR_DEBUG] Top-5 results for query='%s' (master='%s'): %s",
        query, master_key,
        [(name, round(score, 1)) for name, score in scored[:5]],
    )
    _color_debug_logger.debug(
        "[COLOR_DEBUG] WINNER for query='%s': '%s' (score=%.1f, len=%d)",
        query, scored[0][0], scored[0][1], len(scored[0][0]),
    )

    all_matches = [(name, score, 0) for name, score in scored[:10]]


    if not all_matches:
        return empty_result

    # Top match
    top_name, top_score, _ = all_matches[0]
    top_confidence = _score_to_confidence(top_score)
    top_entity = name_to_entity[top_name]

    # Determine matched_id based on confidence tier
    medium_low_score = settings.CONFIDENCE_MEDIUM_LOW * 100  # 60
    matched_id: Optional[int] = None
    matched_label: Optional[str] = None
    extra_data: Dict[str, Any] = {}

    if top_score >= medium_low_score:
        matched_id = int(top_entity["id"])
        matched_label = _get_entity_label(top_entity)
        extra_data = {k: v for k, v in top_entity.items() if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")}
    else:
        # LOW confidence — return alternatives only
        matched_id = None
        matched_label = None

    # Collect up to 2 alternatives above 60%, excluding the top match
    alternatives: List[AlternativeResult] = []
    for alt_name, alt_score, _ in all_matches[1:]:
        if alt_score < medium_low_score:
            break
        alt_entity = name_to_entity[alt_name]
        alternatives.append(
            AlternativeResult(
                id=int(alt_entity["id"]),
                label=_get_entity_label(alt_entity),
                confidence=_score_to_confidence(alt_score),
            )
        )
        if len(alternatives) >= 2:
            break

    logger.debug(
        "match_entity: master='%s' client=%d raw='%s' → matched='%s' "
        "confidence=%.2f alternatives=%d",
        master_key, client_id, raw, matched_label, top_confidence, len(alternatives),
    )

    return MatchResult(
        raw=raw,
        matched_id=matched_id,
        matched_label=matched_label,
        confidence=top_confidence,
        alternatives=alternatives,
        master=master_key,
        extra_data=extra_data,
    )


# ── Exact-match-only API (used by the new stepped workflow) ───────────────────

def match_entity_exact(
    raw: Optional[str],
    master_key: str,
    client_id: int,
    parent_id: Optional[int] = None,
) -> MatchResult:
    """
    Strict exact-match entity lookup.  No fuzzy / rapidfuzz scoring is used.

    Algorithm:
        1. Retrieve entity index slice for master_key + client_id.
        2. Apply parent filter if parent_id is given.
        3. If only 1 entity remains after parent filter, return it (confidence 1.0).
        4. Check UNIT_SYNONYMS canonical lookup.
        5. Exact case-insensitive match (including trailing-dot strip).
        6. Plural shortcut: query ending in \"s\" or \"es\" matches the bare stem.
        7. If none of the above match → raise EntityMatchError.

    Args:
        raw: Raw entity name string extracted from OCR (or user-edited text).
             None / empty → raises EntityMatchError immediately.
        master_key: Entity type key (e.g. \"supplier\", \"transport\").
        client_id: ERP client ID — scopes the lookup.
        parent_id: Optional PK of a parent entity for dependent lookups.

    Returns:
        MatchResult: Exact match result; confidence is always 1.0.

    Raises:
        EntityMatchError: If no exact match is found, or if the entity index
                          is unavailable for the given key/client.
    """
    if isinstance(raw, dict):
        raw = raw.get("label") or raw.get("value") or raw.get("name") or (str(raw.get("id")) if raw.get("id") is not None else None)
    elif raw is not None and not isinstance(raw, str):
        raw = str(raw)

    if not raw or not raw.strip():
        raise EntityMatchError(
            f"Field value is empty for master='{master_key}' "
            f"(client_id={client_id}). Cannot match an empty string."
        )

    try:
        entities = get_client_index(master_key, client_id)
    except ClientNotFoundError as exc:
        raise EntityMatchError(str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise EntityMatchError(
            f"Failed to access entity index for master='{master_key}', "
            f"client_id={client_id}: {exc}"
        ) from exc

    # Apply parent filter if supplied
    search_pool = entities
    if parent_id is not None:
        parent_col = settings.DELIVERY_ADDRESS_PARENT_COL
        search_pool = _filter_by_parent(entities, parent_id, parent_col)
        logger.debug(
            "match_entity_exact: parent filter applied master='%s' parent_id=%d → %d/%d entities.",
            master_key, parent_id, len(search_pool), len(entities),
        )

    if not search_pool:
        raise EntityMatchError(
            f"No exact match found for '{raw}' in master '{master_key}' "
            f"(client_id={client_id}): entity pool is empty after parent filter."
        )

    # Single-entity shortcut (e.g. only one delivery address for this supplier)
    if len(search_pool) == 1:
        sole = search_pool[0]
        logger.debug("match_entity_exact: single entity after filter — auto-matched.")
        return MatchResult(
            raw=raw,
            matched_id=int(sole["id"]),
            matched_label=_get_entity_label(sole),
            confidence=1.0,
            alternatives=[],
            master=master_key,
            extra_data={
                k: v for k, v in sole.items()
                if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")
            },
        )

    # Build name → entity mapping (including secondary name columns)
    name_to_entity: Dict[str, Dict[str, Any]] = {}
    for entity in search_pool:
        main = str(entity.get("main_name") or entity.get("name") or "")
        name_to_entity[main] = entity
        for k, v in entity.items():
            if k not in ("id", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id") and isinstance(v, str) and v.strip():
                if v not in name_to_entity:
                    name_to_entity[v] = entity

    query = raw.strip()
    query_clean = query.rstrip(".").strip().lower()

    # ── Unit synonym lookup ────────────────────────────────────────────────────
    if master_key == "unit":
        canonical_unit = _get_canonical_unit(query_clean)
        if canonical_unit:
            for candidate_name, entity in name_to_entity.items():
                if candidate_name.lower() == canonical_unit:
                    logger.debug(
                        "match_entity_exact: unit synonym '%s' → '%s'",
                        query, _get_entity_label(entity),
                    )
                    return MatchResult(
                        raw=raw,
                        matched_id=int(entity["id"]),
                        matched_label=_get_entity_label(entity),
                        confidence=1.0,
                        alternatives=[],
                        master=master_key,
                        extra_data={
                            k: v for k, v in entity.items()
                            if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")
                        },
                    )

    # ── Exact & plural match (case-insensitive) ────────────────────────────────
    for candidate_name, entity in name_to_entity.items():
        cand_lower = candidate_name.lower()
        if cand_lower == query_clean:
            logger.debug(
                "match_entity_exact: exact match master='%s' client=%d raw='%s' → '%s'",
                master_key, client_id, query, candidate_name,
            )
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=1.0,
                alternatives=[],
                master=master_key,
                extra_data={
                    k: v for k, v in entity.items()
                    if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")
                },
            )
        # Plural shortcuts: "meters" → "meter", "boxes" → "box"
        if query_clean.endswith("s") and query_clean[:-1] == cand_lower:
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=1.0,
                alternatives=[],
                master=master_key,
                extra_data={
                    k: v for k, v in entity.items()
                    if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")
                },
            )
        if query_clean.endswith("es") and query_clean[:-2] == cand_lower:
            return MatchResult(
                raw=raw,
                matched_id=int(entity["id"]),
                matched_label=_get_entity_label(entity),
                confidence=1.0,
                alternatives=[],
                master=master_key,
                extra_data={
                    k: v for k, v in entity.items()
                    if k not in ("id", "main_name", settings.DELIVERY_ADDRESS_PARENT_COL, "client_id")
                },
            )

    # No match found — raise so the caller can accumulate and surface the error
    raise EntityMatchError(
        f"No exact match found for '{raw}' in master '{master_key}' "
        f"(client_id={client_id}). "
        f"Please verify the value or correct the OCR text."
    )
