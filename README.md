# Inventory Wastage Monitor - Multi-Agent System

This repository contains a reference implementation to build a multi-agent system that monitors inventory wastage,
generates recommendations with Gemini, stores/retrieves semantic context from Pinecone, reads source data from Snowflake,
orchestrates agents with LangGraph, and provides a Streamlit UI for review and email delivery to the Warehouse Manager.

## Repo layout
- `streamlit_app.py` — Streamlit UI and manual trigger for analysis
- `snowflake_read.py` — Snowflake read utility
- `pinecone_index.py` — Pinecone upsert/query utilities
- `genai_agent.py` — Gemini (GenAI) embedding & chat wrappers (placeholder — adapt to SDK)
- `email_sender.py` — SMTP / SendGrid email helpers
- `langgraph.yaml` — Example LangGraph orchestration graph
- `.env.example` — Environment variable template (DO NOT COMMIT SECRETS)
- `requirements.txt` — Python package requirements
- `run_steps.md` — Runbook & deployment steps

## Quickstart (local)
1. Copy `.env.example` → `.env` and fill credentials.
2. Create a Python venv: `python -m venv .venv && source .venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Run locally: `streamlit run streamlit_app.py`

## Notes
- Replace placeholder model names and client calls in `genai_agent.py` according to your Gemini SDK.
- Do not store secrets in Git. Use environment-specific secret managers in production.