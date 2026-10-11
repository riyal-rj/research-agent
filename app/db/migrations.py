import hashlib
import time
import asyncpg

from asyncpg.pool import PoolConnectionProxy
from typing import TypeAlias

from app.core.config import settings_inst, Settings
from app.db.pool import get_pool 
from app.core.logging_config import setup_logging
from app.llm.emebeddings import EMBEDDING_DIM

logger = setup_logging(__name__)

MIGRATION_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"research-agent:reports-schema").digest()[:8],
    "big",
    signed=True
)

DDL_LOCK_TIMEOUT = "5s"

EMBEDDING_INDEX = "reports_embeddings_idx"

Conn: TypeAlias = asyncpg.Connection | PoolConnectionProxy

def _table_ddl(dim: int) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS reports(
            id          TEXT PRIMARY KEY,
            topic       TEXT NOT NULL,
            report      TEXT NOT NULL,
            embedding   VECTOR({dim}),
            created_at  TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """

def _index_ddl(lists: int) -> dict[str, str]:
    return {
        EMBEDDING_INDEX : (
            f"CREATE INDEX IF NOT EXISTS {EMBEDDING_INDEX} ON reports"
            f"USING ivfflat (embedding vector_cosine_ops) WITH (lists={lists})"
        ),
        "reports_topic_idx" : "CREATE INDEX IF NOT EXISTS reports_topic_idx ON reports (topic)",
        "reports_created_idx" : (
            "CREATE INDEX IF NOT EXISTS reports_created_idx ON reports (created_at DESC)"
        )
    }

def _forward_server_notice(_connection: object,
                           message: asyncpg.PostgresLogMessage) -> None:
    severity = getattr(message, "severity", None) or "NOTICE"
    text = getattr(message,"message", None) or ""
    extra = " ".join(
        part for part in (
            getattr(message, 'detail', None),
            getattr(message, 'hint', None)
        )
        if part
    )
    logger.warning(
        "PostgreSQL %s during mirgration: %s %s",
        severity,
        text,
        extra
    )

async def _create_extension(conn:Conn) -> None:
    try:
        await conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )
    except asyncpg.InsufficientPrivilegeError as exc:
        raise RuntimeError(
            "pgvector is not installed in this databases and the application role cannot"
            "install it; run `CREATE EXTENSION pgvector` once as a superuser (rds_superuser on RDS)"
        )from exc

async def _check_emebedding_dim(conn: Conn,
                                dim: int) -> None:
    actual = await conn.fetchval(
        """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'reports::regclass 
            AND attname = 'embedding' 
            AND NOT attisdropped
        """
    )
    if actual != dim:
        raise RuntimeError(
            f"reports.embedding is vector({actual}) but EMBEDDING_DIM is {dim}; every insert"
            "and similarity query would fall. Migrate the column (and re-embed) before depploying."
        )


async def _warn_on_lists_draft(conn:Conn,
                               lists:int) -> None:
    options = await conn.fetchval(
        """
            SELECT reloptions
            FROM pg_class
            WHERE oid = to_regclass($1)
        """,
        EMBEDDING_INDEX
    )
    built = next(
        (int(opt.split("=",1)[1]) for opt in options or [] 
        if opt.startswith("lists=")),
        None
    )
    if built is not None and built != lists:
        logger.warning(
            "%s was built with lists=%d but IVFFLAT_LISTS=%d; the settings only applies" \
            "when the index is created. REINDEX INDEX concurrently %s to apply it.",
            EMBEDDING_INDEX,
            built,
            lists,
            EMBEDDING_INDEX
        )

async def db_migrate(config:Settings | None = None) -> None:
    """
        Create the reports schema if misin; verify if it is present.
    """

    dim = int(EMBEDDING_DIM)
    lists = config.ivfflat_lists if config is not None else None
    started = time.perf_counter()
    created: list[str] = []

    async with get_pool().acquire() as connection:
        connection.add_log_listener(_forward_server_notice)
        try:
            async with connection.transaction():
                await connection.execute(
                    """
                        SELECT pg_advisory_xact_lock($1)
                    """,
                    MIGRATION_LOCK_KEY
                )
                waited = time.perf_counter() - started
                if waited > 1.0:
                    logger.info(
                        "Waited %.1fs for another task's schema migration",
                        waited
                    )
                await connection.execute(
                    """
                        SELECT extrversion 
                        FROM pg_extension
                        WHERE extname = "vector"
                    """
                )
