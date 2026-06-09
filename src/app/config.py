from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    root_dir: Path
    provider: str
    model: str
    raw_model: str
    temperature: float
    policy_path: Path
    orders_path: Path
    chroma_dir: Path
    traces_dir: Path
    embedding_model_name: str
    top_k: int
    google_api_key: str | None
    openai_api_key: str | None
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_site_url: str | None
    openrouter_app_name: str | None
    ollama_base_url: str
    custom_llm_base_url: str | None
    custom_llm_api_key: str | None
    custom_llm_model: str | None

    @classmethod
    def load(cls) -> "Settings":
        root_dir = Path(__file__).resolve().parents[2]
        load_dotenv(root_dir / ".env")

        raw_model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        model = raw_model
        provider = os.getenv("LLM_PROVIDER") or _infer_provider(model)

        return cls(
            root_dir=root_dir,
            provider=provider,
            model=model,
            raw_model=raw_model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            policy_path=root_dir / "data" / "policy_mock_vi.md",
            orders_path=root_dir / "data" / "order_customer_mock_data.json",
            chroma_dir=root_dir / "src" / ".chroma",
            traces_dir=root_dir / "src" / "artifacts" / "traces",
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            google_api_key=_clean_secret(os.getenv("GOOGLE_API_KEY")),
            openai_api_key=_clean_secret(os.getenv("OPENAI_API_KEY")),
            openrouter_api_key=_clean_secret(os.getenv("OPENROUTER_API_KEY")),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            custom_llm_base_url=os.getenv("CUSTOM_LLM_BASE_URL"),
            custom_llm_api_key=_clean_secret(os.getenv("CUSTOM_LLM_API_KEY")),
            custom_llm_model=os.getenv("CUSTOM_LLM_MODEL"),
        )


def _infer_provider(model: str) -> str:
    normalized = model.lower()
    if normalized.startswith("gemini"):
        return "gemini"
    if normalized.startswith("gpt") or normalized.startswith("o1") or normalized.startswith("o3"):
        return "openai"
    return "custom"


def _clean_secret(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped in {"your_key_here", "your-api-key", "changeme"}:
        return None
    return stripped
