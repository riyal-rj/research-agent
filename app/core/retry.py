import asyncio
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

async def with_retry(core_fn,
                     max_retries: int = 3,
                     delay:float=1.0,
                     backoff:float=2.0):
    """
        Call core_fn() with exponential backoff. Raises
        the last exception if all the retriues fail.
    """

    last_exc = None
    wait = delay
    for attempt in range(1, max_retries + 1):
        try:
            return await core_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed: {exc}"
                    f"Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                wait *= backoff

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("with_retry did not execute any attempts")