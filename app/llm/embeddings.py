import asyncio
import logging
import threading
import time
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model : SentenceTransformer | None = None
_model_lock = threading.Lock()

def _load() -> SentenceTransformer:
    logger.info(
        "Loading the embedding model %s",
        EMBEDDING_MODEL
    )
    started = time.perf_counter()
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        dim = len(model.encode("warm-up"))
    except Exception:
        logger.exception(
            "Failed to load embedding model %s",
            EMBEDDING_MODEL
        )
        raise
    if dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} produces {dim}-d vectors but EMBEDDING_DIM is {EMBEDDING_DIM};"
            "change both together (reports.embedding is created with EMBEDDING_DIM)."
        )
    logger.info(
        "Embedding model %s ready %d-d on %s, torch threads=%d, %.0f ms",
        EMBEDDING_MODEL,
        dim,
        model.device,
        torch.get_num_threads(),
        (time.perf_counter() - started) * 1000
    )
    return model

def get_model()-> SentenceTransformer:
    """
        Process wide model, ;loaded excatly once ever under even concurrent first calls.
    """
    global _model
    model = _model
    if model is not None:
        return model
    with _model_lock:   
        if _model is None:
            _model = _load()
    return _model


async def load_model() -> None:
    """
        Startup hook: load and verify the model off event loop.
    """
    await asyncio.to_thread(get_model)

async def embed(text: str) -> list[float]:
    return get_model().encode(text, convert_to_numpy=True).tolist()

def cosine_similarity(a:Sequence[float],
                      b:Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype = np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
