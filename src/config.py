"""
Runway Sentinel AI — Centralised Configuration & Fail-Fast Startup.

All other src/ modules import from here. If GROQ_API_KEY is missing
and RUNWAYGUARD_MOCK is not set, this module raises immediately at
import time so misconfigured deployments are caught instantly.

Dual key-resolution strategy:
  1. Read os.environ ("GROQ_API_KEY")  — local .env / Dockerfile ENV
  2. Fall back to st.secrets            — Streamlit Community Cloud
"""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runwaysentinel.config")

# ── Demo / CI override ─────────────────────────────────────────────────────────
# Set RUNWAYGUARD_MOCK=true to run the full UI with synthetic data.
# Useful for Streamlit Community Cloud demo mode and offline CI runs.
USE_MOCK: bool = os.getenv("RUNWAYGUARD_MOCK", "false").lower() in ("true", "1", "yes")

# ── Model selection (overridable via env) ──────────────────────────────────────
VISION_MODEL: str = os.getenv(
    "VISION_MODEL", "llama-3.2-11b-vision-preview"
)
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama3-70b-8192")
REPORT_MODEL: str = os.getenv("REPORT_MODEL", "llama3-70b-8192")


def _get_api_key() -> str:
    """
    Resolve GROQ_API_KEY from environment variables or Streamlit secrets.

    Resolution order:
      1. os.environ["GROQ_API_KEY"]   — always wins (local .env, Dockerfile)
      2. st.secrets["GROQ_API_KEY"]   — Streamlit Community Cloud dashboard
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415  (lazy import intentional)

            key = (st.secrets.get("GROQ_API_KEY") or "").strip()
        except Exception:
            # Streamlit not installed, or called outside a Streamlit process
            pass
    return key


# ── API Key — fail fast ────────────────────────────────────────────────────────
if USE_MOCK:
    GROQ_API_KEY: str = "mock-key-not-used"
    logger.warning(
        "Runway Sentinel AI is running in MOCK mode (RUNWAYGUARD_MOCK=true). "
        "No real Groq API calls will be made."
    )
else:
    GROQ_API_KEY = _get_api_key()
    if not GROQ_API_KEY:
        raise ValueError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  CRITICAL: GROQ_API_KEY is not set!                         ║\n"
            "║  Option 1: Create a .env file with GROQ_API_KEY=gsk_...     ║\n"
            "║  Option 2: Add it to Streamlit Cloud Secrets Manager.        ║\n"
            "║  Option 3: Set RUNWAYGUARD_MOCK=true for offline demo mode.  ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )
    logger.info(
        "Config loaded | Vision: %s | Agent: %s | Report: %s",
        VISION_MODEL,
        AGENT_MODEL,
        REPORT_MODEL,
    )
