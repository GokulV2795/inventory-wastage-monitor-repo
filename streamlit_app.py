import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from snowflake_read import fetch_recent_wastage
from genai_agent import generate_wastage_predictions
from email_sender import send_email_smtp
import tempfile
st.set_page_config(page_title='Inventory Wastage Monitor', layout='wide')
st.title('Inventory Wastage Predictor & Notifier')

days = st.sidebar.slider('Days to analyze', 1, 30, 7)
if st.button('Run analysis now'):
    # Fetch recent data to index and generate predictions (use_gemini=True if you want LLM explanations; set False to skip)
    to_index = fetch_recent_wastage(days)
    if to_index is None or (hasattr(to_index, "empty") and to_index.empty):
        st.warning("No recent inventory data returned; adjust 'Days to analyze' or check data source.")
    else:
        use_gemini = True  # or False if not configured
        pred_df = generate_wastage_predictions(to_index, use_gemini=use_gemini)

        # Show to user
        st.subheader("Predicted Wastage Risk (preview)")
        try:
            st.dataframe(pred_df[["SKU_ID","SKU_NAME","CATEGORY","WASTAGE_PREDICTOR","RISK_SCORE","DAYS_TO_EXPIRY_EST","AI_RECOMMENDATION"]].head(50))
        except Exception:
            st.dataframe(pred_df.head(50))

        # Prepare email content and attachment (CSV)
        manager_email = os.getenv("WAREHOUSE_MANAGER_EMAIL")
        subject = "Inventory Wastage Monitor — Predicted Risks & Recommendations"
        body_lines = [
            "Hello,",
            "",
            "Attached are the latest wastage risk predictions and recommendations.",
            "",
            f"Analysis period: Last {days} days",
            "",
            "Top HIGH risk items:",
        ]
        # summarize top high risk
        top_high = pred_df[pred_df["WASTAGE_PREDICTOR"]=="HIGH"].sort_values("RISK_SCORE", ascending=False).head(10)
        if len(top_high)>0:
            for _, r in top_high.iterrows():
                body_lines.append(f"- {r.get('SKU_NAME') or r.get('SKU_ID')} | Category: {r.get('CATEGORY')} | Risk: {r.get('RISK_SCORE')} | Days to expiry: {r.get('DAYS_TO_EXPIRY_EST')}")
        else:
            body_lines.append("None")

        body_lines.append("")
        body_lines.append("Automated recommendations (sample):")
        # include sample LLM recommendations for top 3 if present
        sample_recs = ( pred_df["AI_RECOMMENDATION"].dropna().astype(str)if isinstance(pred_df, pd.DataFrame) and "AI_RECOMMENDATION" in pred_df.columns else pd.Series([], dtype=str))
        for rec in (sample_recs.head(3).tolist() if hasattr(sample_recs, "head") else list(sample_recs)[:3]):
            body_lines.append(rec)
        body_lines.append("")
        body_lines.append("Regards,")
        body_lines.append("Inventory Wastage Monitor")

        body = "\n".join(body_lines)

        # write CSV to temp file and send
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            temp_csv_path = tf.name
            pred_df.to_csv(temp_csv_path, index=False)

        try:
            if not manager_email:
                st.warning("WAREHOUSE_MANAGER_EMAIL not set; skipping email send.")
            else:
                send_email_smtp(subject=subject, body=body, to=manager_email, attachments=[temp_csv_path])
                st.success(f"Email sent to {manager_email}")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
else:
    st.info("Adjust parameters and click 'Run analysis now' to generate wastage predictions.")