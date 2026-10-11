import asyncio
import logging
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.core.retry import describe_error, with_retry

logger = setup_logging(__name__)

DEFAULT_TIMEOUT = 120

CONNECT_TIMEOUT = 10.0
MAX_BACKOFF_SECONDS = 30.0

RETRYABLE_STATUS = frozenset({
    408,
    429,
    500,
    502,
    503,
    504
})

_client : httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None  = None


class TensorZeroResponseError(ValueError):
    """
        The gateway answered 2XX but the body has no usabke text.
    """

def _get_client() -> httpx.AsyncClient:
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                DEFAULT_TIMEOUT,
                connect=CONNECT_TIMEOUT
            )
        )
        _client_loop = loop

    return _client


async def close_client() -> None:
    """
        Close the shared client (call from the API lifespan / worker shutdown).
    """
    global _client, _client_loop
    client, loop = _client, _client_loop
    _client, _client_loop = None, None
    if client is None or client.is_closed:
        return
    if loop is asyncio.get_running_loop():
        await client.aclose()

def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError) and not isinstance(exc, httpx.UnsupportedProtocol)

def _raise_for_status(response: httpx.Response,
                      function_name: str) -> None:
    if response.is_success:
        return
    try:
        error = response.json().get("error")
        detail = error.get("message") if isinstance(error, dict) else error
    except ValueError:
        detail = None
    detail = str(detail or response.text or response.reason_phrase)[:500]
    raise httpx.HTTPStatusError(
        f"TensorZero {function_name!r} returned {response.status_code} : {detail}",
        request= response.request,
        response=response
    )

def _extract_text(data: dict[str, Any],
                  function_name: str) -> str:
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                return block["text"]
        types  = [block.get("type") for block in content if isinstance(block, dict)]
        raise TensorZeroResponseError(
            f"TensorZero {function_name!r} returned no text block (block_types: {types})"
        )
    output = data.get("output")
    if isinstance(output, dict) and isinstance(output.get("raw"), str):
        return output["raw"] # JSON function : the model's raw JSON text.
    raise TensorZeroResponseError(f"TensorZero {function_name!r} returned neither content nor output")

async def _infer_once(config: Settings,
                      function_name : str,
                      message: str,
                      timeout: float) -> str:
    started = time.perf_counter()
    response = await _get_client().post(
        f"{config.tensorzero_url.rstrip('/')}/inference",
        json = {
            "function_name": function_name,
            "input":{
                "messages":[
                    {
                        "role":"user",
                        "content" : message
                    }
                ]
            }
        },
        timeout = httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT, timeout)),
    )
    _raise_for_status(response, function_name)
    data = response.json()
    text = _extract_text(data, function_name)
    usage = data.get("usage") or {}
    logger.info(
        "TensorZero %s: variant=%s inference_id=%s %.0f ms, tokens in =%s out=%s",
        function_name,
        data.get("variant_name"),
        data.get("inference_id"),
        (time.perf_counter() - started) * 1000,
        usage.get("input_tokens"),
        usage.get("output_tokens")
    )
    return text


async def infer(config: Settings,
                function_name: str,
                message: str,
                timeout: float = DEFAULT_TIMEOUT) -> str:
    """
        Call the TensorZero function with retry and return the text of 
        its first text block.
    """
    try:
        return await with_retry(
            lambda: _infer_once(
                config,
                function_name,
                message,
                timeout
            ),
            max_retries=config.llm_max_retries,
            delay=config.llm_retry_delay,
            retry_on=_is_retryable,
            jitter=True,
            max_delay=MAX_BACKOFF_SECONDS,
            label=f"TensorZero {function_name!r}"
        )
    except Exception as exc:
        logger.error(
            "TensorZero %s failed: %s",
            function_name,
            describe_error(exc)
        )
        raise


