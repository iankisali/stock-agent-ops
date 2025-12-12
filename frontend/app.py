import streamlit as st
import requests
import pandas as pd
import time
import os
import uuid
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Stock Analyst", layout="wide")

# Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("🤖 AI Agent Stock Analyst")
st.markdown(
    f"Enterprise Agentic Workflow Demo "
    f"(Session: `{st.session_state.session_id[:8]}`)"
)

# ---------------------- INPUT UI -------------------------
col1, col2 = st.columns([1, 4])
with col1:
    ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
    use_fmi = st.toggle("Use Finnhub News?", value=True)
    run_btn = st.button("Generate Report", type="primary")

# ---------------------------------------------------------
def plot_prediction_chart(preds, history, ticker):
    """Unified price chart."""
    hist_df = pd.DataFrame(history) if history else pd.DataFrame()
    pred_df = pd.DataFrame(preds)

    # Fix dates
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df["type"] = "Historical"

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["type"] = "Forecast"

    # Connect history → forecast visually
    if not hist_df.empty:
        last_hist = hist_df.iloc[-1]
        connector = pd.DataFrame([{
            "date": last_hist["date"],
            "close": last_hist["close"],
            "type": "Forecast"
        }])
        pred_df = pd.concat([connector, pred_df], ignore_index=True)

    # Plot
    fig = go.Figure()

    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["date"],
            y=hist_df["close"],
            mode="lines",
            name="Historical",
            line=dict(color="#ff4b4b", width=2)
        ))

    fig.add_trace(go.Scatter(
        x=pred_df["date"],
        y=pred_df["close"],
        mode="lines",
        name="Forecast",
        line=dict(color="#00c0f2", width=2)
    ))

    fig.update_layout(
        title=f"{ticker} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
#                        MAIN LOGIC
# ---------------------------------------------------------
if run_btn:
    if not ticker:
        st.error("Ticker is required.")
        st.stop()

    st.markdown("### 1. Running AI Agentic Workflow...")

    with st.spinner("Fetching predictions + news + generating report..."):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json={
                    "ticker": ticker,
                    "use_fmi": use_fmi,
                    "thread_id": st.session_state.session_id
                }
            )

            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
                st.stop()

            data = response.json()

        except Exception as e:
            st.error(f"Connection error: {e}")
            st.stop()

    # -----------------------------------------------------
    # 2. VISUALIZATION SECTION
    # -----------------------------------------------------
    predictions_raw = data.get("predictions", "")
    forecast = []
    history = []

    # Extract forecast list cleanly
    for line in predictions_raw.split("\n"):
        if ":" in line and "$" in line:
            try:
                date_str = line.split(":")[0].strip()
                price = float(line.split("$")[-1])
                forecast.append({"date": date_str, "close": price})
            except:
                pass

    if forecast:
        plot_prediction_chart(forecast, history, ticker)
    else:
        st.warning("No forecast data available to visualize.")

    # -----------------------------------------------------
    # 3. FINAL REPORT
    # -----------------------------------------------------
    st.divider()
    st.markdown("### 2. AI Market Intelligence Report")

    st.markdown(data.get("report", "_No report generated._"))

    # ---------------- Sidebar -----------------
st.sidebar.markdown("### Architecture")
st.sidebar.info("""
**Frontend:** Streamlit  
**Backend:** FastAPI  
**Agent Engine:** LangGraph (Ollama / Mock)  
**Workflows:** Prediction → Sentiment → Report  
""")
