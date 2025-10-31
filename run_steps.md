## Runbook & Steps

1. Provision Pinecone, Google Generative AI access, and Snowflake.
2. Create a Python virtual environment and install requirements.
3. Copy `.env.example` to `.env` and fill in secrets (DO NOT commit `.env`).
4. Validate:
   - `python -c "from snowflake_read import fetch_recent_wastage; print(fetch_recent_wastage(1).head())"`
   - Test embeddings via `genai_agent.embed_text('test')`
   - Test Pinecone upsert/query via `pinecone_index.upsert_chunks(...)` and `query_similar(...)`
5. Run Streamlit: `streamlit run streamlit_app.py`
6. Deploy LangGraph graph and wire triggers if using LangGraph hosted runner.