from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 127.0.0.1, not "localhost": on this Windows setup, resolving the
    # "localhost" hostname through getaddrinfo() can take over two minutes
    # (IPv6/VPN interference) even though the IPv4 address answers instantly.
    database_url: str = "postgresql://maintidoc:maintidoc@127.0.0.1:5432/maintidoc"

    llm_provider: Literal["ollama", "openrouter"] = "ollama"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "qwen2.5:7b-instruct"
    ollama_vlm_model: str = "qwen2.5vl:3b"
    # Keeps the text LLM loaded in VRAM between questions (Ollama's own
    # default is 5 minutes) -- avoids a multi-second cold reload if the user
    # pauses between chat turns during a demo.
    ollama_keep_alive: str = "30m"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_llm_model: str = "qwen/qwen-2.5-7b-instruct"
    openrouter_vlm_model: str = "qwen/qwen-2.5-vl-3b-instruct"

    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16
    embedding_max_length: int = 1024

    clip_model: str = "MobileCLIP-S2"
    clip_pretrained: str = "datacompdr"

    # "auto" picks cuda if available else cpu (logged, never silent); "cuda"/"cpu"
    # force that choice. embed_query_device overrides device for single-query
    # BGE-M3 embeddings only (e.g. run on CPU while the GPU holds the LLM).
    device: Literal["auto", "cuda", "cpu"] = "auto"
    embed_query_device: Literal["auto", "cuda", "cpu"] | None = None

    pgvector_text_table: str = "maintidoc_text"
    pgvector_image_table: str = "maintidoc_images"
    pgvector_text_dim: int = 1024
    pgvector_image_dim: int = 512
    pgvector_hybrid_search: bool = True
    pgvector_text_search_config: str = "french"

    rag_k: int = 8
    rag_max_nodes: int = 6
    rag_context_budget_tokens: int = 2500
    rag_min_score: float = 0.50
    rag_temperature: float = 0.1
    rag_num_predict: int = 600
    rag_num_ctx: int = 8192
    rag_query_expansion: bool = True
    rag_rrf_k: int = 60
    # Skip the (costly) expand_query LLM call when the original question's
    # own best_score is already this confident -- calibrated on the gold set,
    # see docs/05_rapport_rag.md section 0c.
    rag_expansion_skip_score: float = 0.65

    # Image search rejection threshold (M7.4 section 3): calibrated on real
    # data (51 perturbed manual figures vs. 6 locally-generated unrelated
    # images -- see scripts/_calibrate_image_threshold.py), in-domain top1
    # scores 0.78-0.99 vs. out-of-domain 0.40-0.71. 0.78 gives 100%
    # out-of-domain rejection while keeping in-domain recall@3 at 96.1%
    # (>= the 95% target). Never guessed.
    image_min_score: float = 0.78

    agent_max_tool_calls: int = 4
    agent_timeout_seconds: float = 90.0

    media_dir: Path = Path("data/media")
    pdf_path: Path = Path("data/pdf/document.pdf")
    models_dir: Path = Path("models")

    @model_validator(mode="after")
    def _guard_openrouter_key(self) -> "Settings":
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set "
                "(get one at https://openrouter.ai/keys)."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
