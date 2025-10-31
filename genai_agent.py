# genai_agent.py
import os
import math
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Optional import for Gemini (Google generative AI) SDK
try:
    from google import generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Read Gemini API key from env if needed (provider-specific)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



def _norm_cols_upper(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip().upper() for c in df2.columns]
    return df2

def _parse_date_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def compute_wastage_predictor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple, explainable wastage risk score and label using:
      - DAILY_SALES_RATE
      - SHELF_LIFE_DAYS
      - EXPIRY_DATE
      - OPENING_INVENTORY
      - CURRENT INVENTORY or similar if exists (try columns OPENING_INVENTORY, ON_HAND, AVAILABLE)
    Returns df with new columns: RISK_SCORE (0-100), Wastage_Predictor (LOW/MEDIUM/HIGH)
    """
    df = _norm_cols_upper(df)
    out = df.copy()

    # Try to detect inventory qty column candidates
    qty_cols = [c for c in ["OPENING_INVENTORY","ON_HAND","AVAILABLE_QTY","AVAILABLE","CURRENT_QTY","QTY"] if c in out.columns]
    qty_col = qty_cols[0] if qty_cols else None

    # Columns we will use (if missing, assume safe defaults)
    dsr_col = "DAILY_SALES_RATE" if "DAILY_SALES_RATE" in out.columns else None
    shelf_col = "SHELF_LIFE_DAYS" if "SHELF_LIFE_DAYS" in out.columns else None
    expiry_col = "EXPIRY_DATE" if "EXPIRY_DATE" in out.columns else None

    # compute numeric safe getters
    def _num(x):
        try:
            if pd.isna(x): return float("nan")
            return float(x)
        except Exception:
            return float("nan")

    risk_scores = []
    predictors = []
    days_to_expiry_list = []
    today = datetime.utcnow().date()

    for _, row in out.iterrows():
        dsr = _num(row.get(dsr_col)) if dsr_col else float("nan")
        shelf = _num(row.get(shelf_col)) if shelf_col else float("nan")
        qty = _num(row.get(qty_col)) if qty_col else float("nan")
        expiry = _parse_date_safe(row.get(expiry_col)) if expiry_col else None

        # days to expiry
        if expiry:
            days_to_expiry = (expiry - today).days
        else:
            # fallback: estimate from shelf life and date columns if present
            days_to_expiry = None

        days_to_expiry_list.append(days_to_expiry)

        # Heuristic scoring rules (explainable):
        # Start with 0 score. Increase score (higher -> more risk) when:
        #  - days_to_expiry is low (<= 7 days: +40)
        #  - days_to_expiry between 8-30: +20
        #  - low daily sales relative to inventory (inventory / daily_sales_rate gives days of cover)
        #  - if days_of_cover > shelf_life or > 30 days, risk increases.
        score = 0.0

        # expiry contribution
        if days_to_expiry is not None:
            if days_to_expiry <= 0:
                score += 50
            elif days_to_expiry <= 7:
                score += 40
            elif days_to_expiry <= 30:
                score += 20
            elif days_to_expiry <= 90:
                score += 5

        # inventory vs sales
        days_of_cover = None
        if not math.isnan(dsr) and dsr > 0 and not math.isnan(qty):
            days_of_cover = qty / dsr
            # if cover exceeds shelf life, risk of wastage
            if not math.isnan(shelf):
                if days_of_cover > shelf:
                    score += 30
                elif days_of_cover > shelf * 0.8:
                    score += 10
            else:
                # generic thresholds
                if days_of_cover > 30:
                    score += 20
                elif days_of_cover > 14:
                    score += 10

            # low sales -> higher risk when expiry is near
            if days_to_expiry is not None and days_of_cover > days_to_expiry and days_to_expiry <= 30:
                score += 20

        # shelf life very short with moderate stock
        if not math.isnan(shelf) and shelf <= 7 and (not math.isnan(qty) and qty > 0):
            score += 15

        # cap score 0-100
        score = max(0, min(100, score))
        risk_scores.append(round(score, 2))

        # map to categories
        if score >= 60:
            predictors.append("HIGH")
        elif score >= 30:
            predictors.append("MEDIUM")
        else:
            predictors.append("LOW")

    out["RISK_SCORE"] = risk_scores
    out["WASTAGE_PREDICTOR"] = predictors
    out["DAYS_TO_EXPIRY_EST"] = days_to_expiry_list
    return out

def _build_system_prompt():
    """
    System prompt to instruct Gemini/LLM to behave as your Wastage Risk Prevention Assistant.
    """
    return (
        "You are a Wastage Risk Prevention Assistant for warehouse operations.\n"
        "Use the variables: {{Warehouse}} {{Category}} {{SKU_Name}} {{Wastage_Predictor}}\n"
        "From this {{Wastage_Predictor}} dataset utilize the columns (Daily Sales Rate and respective columns) "
        "and provide concise, actionable recommendations for warehouse managers. "
        "Be specific (what to do, quantities where relevant, whether to transfer, markdown, discount, or push-promotions). "
        "Keep responses short (3 recommendations) and include one immediate action and one monitoring action."
    )

def call_gemini_for_explanations(rows: list, timeout: int = 20) -> list:
    """
    Enhanced LLM recommendation generator for wastage prevention.
    Auto-detects available Gemini model (v1 or v1beta).
    """
    if not GENAI_AVAILABLE or not GOOGLE_API_KEY:
        # Fallback logic (same as before)
        suggestions = []
        for r in rows:
            predictor = str(r.get("WASTAGE_PREDICTOR", "")).upper()
            sku = r.get("SKU_NAME", "") or r.get("SKU_ID", "")
            wh = r.get("Warehouse", "")
            days_to_expiry = r.get("DAYS_TO_EXPIRY_EST")
            dsr = r.get("DAILY_SALES_RATE")
            if predictor == "HIGH":
                suggestion = (
                    f"[High Risk] SKU '{sku}' at {wh}: nearing expiry. "
                    f"Run promotions or markdowns to clear stock within {days_to_expiry} days. "
                    f"Transfer to high-demand locations if DSR={dsr} is low."
                )
            elif predictor == "MEDIUM":
                suggestion = (
                    f"[Medium Risk] SKU '{sku}' at {wh}: monitor weekly. "
                    f"Adjust replenishment to prevent overstock."
                )
            else:
                suggestion = (
                    f"[Low Risk] SKU '{sku}' at {wh}: maintain FIFO and current sales pace."
                )
            suggestions.append(suggestion)
        return suggestions

    # ---- Initialize Gemini ----
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

    # Try flash first, then fallback to gemini-pro if unavailable
    model_name = "gemini-2.5-pro"
    try:
        model = genai.GenerativeModel(model_name)
        _ = model.generate_content("Test connection")  # sanity check
    except Exception:
        model_name = "models/gemini-2.5-pro"
        model = genai.GenerativeModel(model_name)

    print(f"[INFO] Using Gemini model: {model_name}")

    system_prompt = (
        "You are an AI Wastage Prevention Assistant for warehouse operations.\n"
        "For each SKU, generate 3 short, practical actions to PREVENT wastage.\n"
        "Be specific: redistribution, discounting, FIFO/FEFO, reorder control, etc."
    )

    results = []
    for r in rows:
        sku = r.get("SKU_NAME", "") or r.get("SKU_ID", "")
        warehouse = r.get("Warehouse", "")
        category = r.get("Category", "")
        predictor = str(r.get("WASTAGE_PREDICTOR", ""))
        expiry = r.get("DAYS_TO_EXPIRY_EST", "")
        dsr = r.get("DAILY_SALES_RATE", "")
        inv = r.get("OPENING_INVENTORY", "")

        user_prompt = (
            f"{system_prompt}\n\n"
            f"Warehouse: {warehouse}\n"
            f"Category: {category}\n"
            f"SKU Name: {sku}\n"
            f"Daily Sales Rate: {dsr}\n"
            f"Opening Inventory: {inv}\n"
            f"Days to Expiry: {expiry}\n"
            f"Wastage Predictor: {predictor}\n\n"
            "Suggest 3 concise, preventive actions to reduce wastage for this SKU."
        )

        try:
            response = model.generate_content(user_prompt)
            text = getattr(response, "text", None)
            if not text and hasattr(response, "candidates"):
                text = response.candidates[0].content.parts[0].text
            results.append(text.strip() if text else "(No response)")
        except Exception as e:
            results.append(f"AI Suggestion unavailable: {e}")

    return results



def generate_wastage_predictions(df: pd.DataFrame, use_gemini: bool = True) -> pd.DataFrame:
    """
    Compute wastage predictor + LLM prevention recommendations.
    AI_RECOMMENDATION column now contains AI-generated or heuristic suggestions
    to STOP wastage (actions like markdown, transfer, etc.).
    """
    df_pred = compute_wastage_predictor(df)
    dfu = df_pred.copy()
    dfu.columns = [c.upper() for c in dfu.columns]

    if use_gemini:
        # Prepare structured rows for Gemini call
        rows = []
        for _, r in dfu.iterrows():
            rows.append({
                "Warehouse": r.get("WAREHOUSE", ""),
                "Category": r.get("CATEGORY", ""),
                "SKU_NAME": r.get("SKU_NAME", "") or r.get("SKU_ID", ""),
                "DAILY_SALES_RATE": r.get("DAILY_SALES_RATE", ""),
                "OPENING_INVENTORY": r.get("OPENING_INVENTORY", ""),
                "DAYS_TO_EXPIRY_EST": r.get("DAYS_TO_EXPIRY_EST", ""),
                "WASTAGE_PREDICTOR": r.get("WASTAGE_PREDICTOR", ""),
            })
        df_pred["AI_RECOMMENDATION"] = call_gemini_for_explanations(rows)
    else:
        df_pred["AI_RECOMMENDATION"] = call_gemini_for_explanations([], timeout=0)

    return df_pred
