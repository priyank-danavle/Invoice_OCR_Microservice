"""
app/db/indexer.py
─────────────────
Loads all ERP master tables from MySQL into an in-memory entity index on
FastAPI startup, and refreshes them on a configurable TTL (default 60 min).

Index structure:
    entity_index[master_key][client_id] = [
        {"id": 1, "name": "Reliance Industries", ...extra_cols},
        ...
    ]

All queries use parameterised WHERE client_id = ? — data is never mixed
across clients.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiomysql

from app.config import settings

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

# entity_index[master_key][client_id] → list of entity dicts
EntityIndex = Dict[str, Dict[int, List[Dict[str, Any]]]]


# ── Module-level state ────────────────────────────────────────────────────────

entity_index: EntityIndex = {}
_is_loaded: bool = False
_last_refreshed: Optional[datetime] = None
_pool: Optional[aiomysql.Pool] = None
_refresh_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]


# ── Custom exceptions ─────────────────────────────────────────────────────────

class ClientNotFoundError(Exception):
    """Raised when a client_id is not present in the entity index."""


class IndexNotLoadedError(Exception):
    """Raised when a match is attempted before the index has been populated."""


# ── Pool lifecycle ────────────────────────────────────────────────────────────

async def create_pool() -> aiomysql.Pool:
    """
    Create and return a new aiomysql connection pool.

    Returns:
        aiomysql.Pool: Ready-to-use async pool.

    Raises:
        aiomysql.Error: If the database is unreachable.
    """
    pool = await aiomysql.create_pool(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        db=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        connect_timeout=settings.DB_READ_TIMEOUT,
        autocommit=True,
        minsize=1,
        maxsize=5,
        charset="utf8mb4",
    )
    logger.info("MySQL connection pool created (host=%s db=%s).", settings.DB_HOST, settings.DB_NAME)
    return pool


async def close_pool() -> None:
    """
    Close the aiomysql connection pool gracefully.

    Called during FastAPI shutdown event to release DB connections.
    """
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("MySQL connection pool closed.")


# ── Core loader ───────────────────────────────────────────────────────────────

async def _load_master_table(
    conn: aiomysql.Connection,
    master_key: str,
    table_cfg: Dict[str, Any],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load one master table from MySQL, grouped by client_id.

    Fetches id, client_id, all columns listed in table_cfg["name_cols"],
    plus any extra columns (e.g. ledger FK for delivery_address).
    The FIRST entry in name_cols is aliased as "name" so the matcher
    can always reference entity["name"] uniformly.
    All other name_cols columns are stored in the entity dict under
    their original DB column names.

    Args:
        conn: An open aiomysql connection.
        master_key: Logical name of the entity type (e.g. "supplier").
        table_cfg: Dict with {"table": ..., "name_cols": [col1, col2, ...]}.

    Returns:
        Dict[int, List[Dict]]: Mapping client_id → list of entity dicts.
    """
    table_name = table_cfg["table"]
    name_cols: List[str] = table_cfg["name_cols"]   # guaranteed non-empty
    primary_col = name_cols[0]                       # first col → "name" alias

    # Build column list:
    #   primary col aliased as "main_name" for uniform matcher access
    #   remaining name_cols fetched under their own names
    col_parts = [f"{primary_col} AS main_name"]
    for col in name_cols[1:]:
        col_parts.append(col)

    is_global = table_cfg.get("is_global", False)
    has_active = table_cfg.get("has_active", True)

    # delivery_address also needs ledger_id for parent-filtered matching.
    extra_col = ""
    if master_key == "delivery_address":
        extra_col = f", {settings.DELIVERY_ADDRESS_PARENT_COL}"

    cols_sql = ", ".join(col_parts)
    
    where_clause = "WHERE active = 1" if has_active else ""
    
    if is_global:
        query = (
            f"SELECT id, {cols_sql} "
            f"FROM `{table_name}` {where_clause}"
        )
    else:
        query = (
            f"SELECT id, {cols_sql}, client_id{extra_col} "
            f"FROM `{table_name}` {where_clause}"
        )

    async with conn.cursor(aiomysql.DictCursor) as cur:
        try:
            await cur.execute(query)
            rows = await cur.fetchall()
        except aiomysql.Error as exc:
            logger.warning(
                "Could not load master table '%s' (table=%s): %s",
                master_key, table_name, exc,
            )
            return {}

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    if is_global:
        grouped[0] = []
        for row in rows:
            grouped[0].append(dict(row))
    else:
        for row in rows:
            cid = int(row["client_id"])
            grouped.setdefault(cid, []).append(dict(row))

    record_count = sum(len(v) for v in grouped.values())
    logger.debug(
        "Loaded master '%s': %d records across %d clients.",
        master_key, record_count, len(grouped),
    )
    return grouped


async def load() -> None:
    """
    Populate the global entity_index from MySQL.

    Connects to MySQL, iterates over all master tables defined in
    settings.MASTER_TABLE_MAP, and stores results grouped by client_id.

    Sets _is_loaded = True and _last_refreshed to now (UTC) when done.

    Raises:
        aiomysql.Error: If database connection fails on first load.
    """
    global entity_index, _is_loaded, _last_refreshed, _pool

    logger.info("Starting entity index load from MySQL …")

    if _pool is None:
        try:
            _pool = await create_pool()
        except aiomysql.Error as exc:
            logger.error("Failed to connect to MySQL for index load: %s", exc)
            raise

    new_index: EntityIndex = {}

    async with _pool.acquire() as conn:
        for master_key, table_cfg in settings.MASTER_TABLE_MAP.items():
            new_index[master_key] = await _load_master_table(conn, master_key, table_cfg)

    entity_index = new_index
    _is_loaded = True
    _last_refreshed = datetime.now(tz=timezone.utc)

    total_clients = len(
        {cid for per_key in entity_index.values() for cid in per_key}
    )
    logger.info(
        "Entity index loaded: %d master types, %d distinct clients. Refreshed at %s.",
        len(entity_index), total_clients, _last_refreshed.isoformat(),
    )


# ── Background refresh ────────────────────────────────────────────────────────

async def _refresh_loop() -> None:
    """
    Background coroutine that reloads the entity index every TTL minutes.

    Runs indefinitely until the FastAPI shutdown event cancels the task.
    Errors during refresh are logged but do not crash the service.
    """
    ttl_seconds = settings.ENTITY_INDEX_TTL_MINUTES * 60
    while True:
        await asyncio.sleep(ttl_seconds)
        try:
            logger.info("TTL expired — refreshing entity index …")
            await load()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Entity index refresh failed: %s", exc)


def start_background_refresh() -> None:
    """
    Schedule the background TTL refresh loop as an asyncio task.

    Must be called after the FastAPI event loop is running (i.e. inside a
    lifespan startup handler or @app.on_event("startup") handler).
    """
    global _refresh_task
    _refresh_task = asyncio.create_task(_refresh_loop())
    logger.info(
        "Entity index background refresh scheduled (TTL=%d min).",
        settings.ENTITY_INDEX_TTL_MINUTES,
    )


def stop_background_refresh() -> None:
    """
    Cancel the background refresh task during FastAPI shutdown.
    """
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        _refresh_task.cancel()
        logger.info("Entity index background refresh task cancelled.")


# ── Status accessors ──────────────────────────────────────────────────────────

def is_loaded() -> bool:
    """Return True once the entity index has been successfully populated."""
    return _is_loaded


def last_refreshed() -> Optional[datetime]:
    """Return the UTC datetime of the last successful index load."""
    return _last_refreshed


def indexed_client_count() -> int:
    """Return the number of distinct client_ids present in the entity index."""
    return len({cid for per_key in entity_index.values() for cid in per_key})


def get_client_index(master_key: str, client_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve the entity list for a specific master key and client.

    Args:
        master_key: Entity type (e.g. "supplier", "transport").
        client_id: ERP client identifier.

    Returns:
        List[Dict]: List of entity dicts for that client.

    Raises:
        IndexNotLoadedError: If load() has not completed yet.
        ClientNotFoundError: If client_id has no records for master_key.
    """
    if not _is_loaded:
        raise IndexNotLoadedError(
            "Entity index is not loaded yet. The service may still be starting up."
        )

    master_data = entity_index.get(master_key, {})
    
    # Check if this is a global table (stored under client_id 0)
    if 0 in master_data:
        return master_data[0]

    if client_id not in master_data:
        logger.warning(
            "client_id=%d not found in entity index for master='%s'.",
            client_id, master_key,
        )
        raise ClientNotFoundError(
            f"No data found for client_id={client_id} in master '{master_key}'. "
            "Ensure this client exists in the database and the index is loaded."
        )
    return master_data[client_id]
