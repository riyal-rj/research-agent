import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

T= TypeVar("T")

def _retry_everything(_exc: Exception) -> bool:
    return True

def describe_error(exc: BaseException) -> str:
    """
        One line `Type: message`; some errors (e.g. httpx.ReadTimeout() have an empty str()
    """
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0]}" if text else type(exc).__name__

async def with_retry(core_fn : Callable[[], Awaitable[T]],
                     max_retries: int = 3,
                     delay:float=1.0,
                     backoff:float=2.0,
                     *,
                     retry_on: Callable[[Exception], bool] = _retry_everything,
                     jitter: bool = False,
                     max_delay: float | None = None,
                     label: str = "operation") -> T:
    """
        Await core_fn() with exponential backoff and return its result.
    """
    attempts = max(1, max_retries)
    wait = delay
    for attempt in range(1, attempts + 1):
        try:
            return await core_fn()
        except Exception as exc:
            if attempt == attempts or not retry_on(exc):
                raise
            sleep_for = wait if max_delay is None else min(wait, max_delay)
            if jitter:
                sleep_for = random.uniform(0.0,sleep_for)
            logger.warning(
                "%s attempt %d/%d failed: %s; retrying in %.1fs",
                label,
                attempt,
                attempts,
                describe_error(exc),
                sleep_for
            )
            await asyncio.sleep(sleep_for)
            wait *= backoff
    raise RuntimeError("with_retry did not execute any attempts.")