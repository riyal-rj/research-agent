import os
import langsmith
from langsmith import utils as ls_utils

from app.core.config import settings_inst, Settings
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

def configure_tracing(settings: Settings | None = None) -> bool:
    if settings is None:
        settings = settings_inst

    if not settings.langsmith_api_key:
        langsmith.configure(enabled=False)
        logger.info(
            f"LangSmith tracing disabled: no API Key configurerd"
        )
        return False

    os.environ.update(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": settings.langsmith_api_key,
            "LANGSMITH_PROJECT": settings.langchain_project,
            "LANGSMITH_ENDPOINT": LANGSMITH_ENDPOINT,
        }
    )

    for memoised in (ls_utils.get_env_var,
                     ls_utils.get_tracer_project):
        cache_clear = getattr(memoised, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    langsmith.configure(
        client=langsmith.Client(
            api_key=settings.langsmith_api_key,
            api_url=LANGSMITH_ENDPOINT
        ),
        enabled=True,
        project_name=settings.langchain_project
    )

    logger.info(
        f"Langsmith tracing enabled (porject=%s)",
        settings.langchain_project
    )

    return True
