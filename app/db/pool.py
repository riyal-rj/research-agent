import asyncpg
import re
import time
from urllib.parse import urlsplit
import asyncio
from app.core.config import settings_inst, Settings
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

APPLICATION_NAME = "research-agent"
CLOSE_TIMEOUT_SECONDS = 10.0
_PG_IDENTIFIER_MAX = 63

_DRIVER_SUFFIX = re.compile(r"postgres(?:ql)?\+\w+://", re.IGNORECASE)

_pool:asyncpg.Pool | None  = None

 
def _asyncpg_dsn(url: str) -> str:
    return _DRIVER_SUFFIX.sub("postgresql://", url, count=1)
 
def _describe(dsn: str) -> str:
    """
        Render user@host:port/db for logs: never the password or query 
        string.
        A password that was not URL-encoded makes the URL ambiguous
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "<redacted: unparsable DSN>"
    if parts.fragment or "@" in parts.path or "@" in parts.query:
        return "<redacted: malformed DSN>"
    userinfo,_,hosts = parts.netloc.partition("@")
    user = userinfo.partition(":")[0]
    target = f"{hosts}{parts.path}"
    return f"{user}@{target}" if user else target

async def init_pool(settings:Settings | None = None) -> asyncpg.Pool:
    """
        Create the process wide pool and verify the database is reachable.
    """
    global _pool
    if _pool is not None:
        logger.warning(
            f"PostgresSQL pool already initailized, reusing it."
        )
        return _pool

    if settings is None:
        settings = settings_inst

    dsn = _asyncpg_dsn(settings.database_url)
    target = _describe(dsn)
    application_name =f"{APPLICATION_NAME}:{settings.consumer_name}"[:_PG_IDENTIFIER_MAX]

    logger.info(
        "Connecting to PostgresSQL %s (pool min=%d, max=%d)",
        target,
        settings.db_pool_min,
        settings.db_pool_max,
    )

    started = time.perf_counter()
    try:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            server_settings={"application_name": application_name},
        )
        try:
            async with pool.acquire() as connection:
                await connection.execute(
                    f"SELECT 1"
                )
                version = connection.get_server_version()
        except BaseException:
            pool.terminate()
            raise
    except Exception as exc:
        logger.error(
            f"PostgresSQL pool intailisation failed for %s: %s: %s",
            target, type(exc).__name__, exc
        )
        raise

    if _pool is not None:
        await pool.close()
        return _pool

    _pool = pool
    logger.info(
        f"PostgresSQL pool ready: %s, server: %d:%d, application_name=%s, %.0f ms",
        target,
        version.major,
        version.minor,
        application_name,
        (time.perf_counter() - started) * 1000
    )
    return pool

async def close_pool(timeout:float = CLOSE_TIMEOUT_SECONDS) -> None:
    """
        Close the pool, waiting at most 'timeout' seconds for checked-out connections.

        Pool.close() waits indefinitely for every connections to be released.
    """
    global _pool
    pool, _pool = _pool, None
    if pool is None:
        return

    logger.info(
        f"Closing PostgresSQL pool"
    )
    try:
        await asyncio.wait_for(pool.close(), timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "PostgresSQL pool did not close within %.1fs; connection terminated",
            timeout
        )
    except Exception:
        logger.error(
            "Error while closing PostgreSQL pool; connection terminated."
        )
    else:
        logger.info(
            "PostgreSQL pool closed."
        )

def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError(
            "Database pool not initialized, call init_pool() first during startup."
        )
    return _pool


