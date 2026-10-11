import os
import socket
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import (
    AWSSecretsManagerSettingsSource,
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict
)

DEFAULT_SECRET_ID = "research-agent/config"
DEFAULT_REGION = "us-east-1"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        coerce_numbers_to_str=True,
    )

    @classmethod
    def settings_customise_sources(cls,
                                   settings_cls : type[BaseSettings],
                                   init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource,
                                   dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource,) -> tuple[PydanticBaseSettingsSource, ...]:
        sources : list[PydanticBaseSettingsSource] = [init_settings]

        secret_id = os.environ.get("CONFIG_SECRET_ID", DEFAULT_SECRET_ID).strip()
        if secret_id:
            sources.append(
                AWSSecretsManagerSettingsSource(
                    settings_cls,
                    secret_id = secret_id,
                    region_name = os.environ.get("AWS_REGION",DEFAULT_REGION),
                    case_sensitive= False   
                )
            )
        sources += [env_settings, dotenv_settings]
        return tuple(sources)

    aws_region : str = Field(
        default=DEFAULT_REGION,
        description="AWs Region for the Bedrock and other ASWS Servcies clients."
    )

    bedrock_guardrails_is : str = Field(
        ...,
        description="Bedrock Guardrail iodentifier applied to identification."
    )

    bedrock_guardrail_version: str = Field(
        ...,
        description="Bedrock Guardrail version (numeric version or draft)"
    )

    redis_url: str = Field(
        ...,
        repr=False,
        description="Redis url for the cache , sessions, rate limiting and job stream."
    )

    database_url: str = Field(
        ...,
        repr=False,
        description="PostgresSQL (pgvector) database url."
    )

    tensorzero_url: str = Field(
        ...,
        description="TensorZero gateway base URL for LLM inference."
    )

    api_key: str = Field(
        default="",
        repr=False,
        description="API Key"
    )


    #Langsmith
    langsmith_api_key: str = Field(
        default="",
        repr=False,
        description="Lanfgsmith API Key"
    )

    langchain_project: str = Field(
        default="research-agent",
        description="LangSmith project that traces are logged to"
    )
 
    langsmith_dataset: str = Field(
        default="research-agent-reports",
        description="LangSmith dataset used for report evaluation"
    )

    
    # Semantic cache
 
    cache_ttl: int = Field(
        default=3600,
        gt=0,
        description="Semantic cache entry TTL in seconds"
    )
 
    cache_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum embedding similarity for a semantic cache hit"
    )

    
    # Short-term session memory
 
    session_ttl: int = Field(
        default=1800,
        gt=0,
        description="Session history TTL in seconds"
    )
 
    session_max_messages: int = Field(
        default=5,
        ge=1,
        description="Maximum messages retained per session"
    )
 
    session_content_truncate: int = Field(
        default=500,
        gt=0,
        description="Character limit per stored session message"
    )

    
    # Long-term memory (pgvector)
 
    ltm_days: int = Field(
        default=7,
        ge=1,
        description="Long-term memory lookback window in days"
    )
 
    ltm_threshold: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for a long-term memory match"
    )
 
    ltm_diff_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for the long-term memory diff lookup"
    )
 
    ltm_diff_limit: int = Field(
        default=5,
        ge=1,
        description="Maximum rows returned by the long-term memory diff lookup"
    )
 
    ivfflat_lists: int = Field(
        default=100,
        ge=1,
        description="Number of lists for the pgvector IVFFlat index"
    )

    
    # Redis Streams job queue
 
    stream_key: str = Field(
        default="research:jobs",
        description="Redis Stream key holding queued research jobs"
    )
 
    consumer_group: str = Field(
        default="workers",
        description="Redis Streams consumer group shared by all workers"
    )
 
    consumer_name: str = Field(
        default_factory=socket.gethostname,
        description="Redis Streams consumer name; hostname is unique per ECS task, "
                    "so consumers stay distinct when scaling out"
    )
 
    result_ttl: int = Field(
        default=3600,
        gt=0,
        description="Seconds a completed job result is kept in Redis"
    )

     # Agent
 
    agent_report_truncate: int = Field(
        default=3000,
        gt=0,
        description="Character limit for reports inside the agent loop"
    )
 
    agent_max_iterations: int = Field(
        default=2,
        ge=1,
        description="Maximum agent loop iterations per job"
    )
 
    # Evaluation
 
    eval_report_truncate: int = Field(
        default=1500,
        gt=0,
        description="Character limit for reports sent to the evaluator"
    )
 
    eval_comment_truncate: int = Field(
        default=300,
        gt=0,
        description="Character limit for evaluator comments"
    )
 
    # LLM resilience
 
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Retry attempts for failed LLM calls"
    )
 
    llm_retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay in seconds between LLM retries"
    )

    # Rate limiting
 
    rate_limit_requests: int = Field(
        default=10,
        ge=1,
        description="Requests allowed per client within one rate-limit window"
    )
 
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Rate-limit window length in seconds"
    )
 
    # Database pool
 
    db_pool_min: int = Field(
        default=2,
        ge=0,
        description="Minimum connections kept in the PostgreSQL pool"
    )
 
    db_pool_max: int = Field(
        default=10,
        ge=1,
        description="Maximum connections in the PostgreSQL pool"
    )
 
    @model_validator(mode="after")
    def _check_db_pool_bounds(self) -> "Settings":
        if self.db_pool_min > self.db_pool_max:
            raise ValueError(
                f"db_pool_min ({self.db_pool_min}) must be <= db_pool_max ({self.db_pool_max})"
            )
        return self
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton: Secrets Manager is called once per process."""
    return Settings()  # type: ignore[call-arg]  # Values are loaded from env / Secrets Manager.
 
 
settings_inst = get_settings()
 
 