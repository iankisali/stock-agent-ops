"""
Agent nodes for: 
- performance analysis
- market sentiment
- final Bloomberg-style report
"""
from datetime import datetime
from langchain_core.messages import SystemMessage, AIMessage

from src.agents.tools import get_stock_predictions, get_stock_news, TOOLS_LIST


# LLM setup (Ollama or Mock fallback)
try:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="gpt-oss:20b-cloud",
        temperature=0.3,
        base_url="http://host.docker.internal:11434"
    ).bind_tools(TOOLS_LIST)

except Exception:
    class Mock:
        def invoke(self, *args, **kwargs):
            return AIMessage(content="Mock response: LLM unavailable.")
    llm = Mock()


# --------------------------------------------------------------------------
# PERFORMANCE ANALYST
# --------------------------------------------------------------------------
def performance_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    
    predictions = state.get("predictions")  # <--- USE what analyze_stock already fetched

    if predictions == "__MODEL_TRAINING__":
        return {
            "messages": [],
            "predictions": "__MODEL_TRAINING__"
        }


# --------------------------------------------------------------------------
# MARKET EXPERT
# --------------------------------------------------------------------------
def market_expert_node(state: dict) -> dict:
    ticker = state["ticker"]
    news = get_stock_news(ticker)

    prompt = f"""
You are a market strategist summarizing sentiment.
News:

{news}

Return a 3–5 line sentiment summary.
"""
    resp = llm.invoke([SystemMessage(content=prompt)])

    return {
        "messages": [resp],
        "news_sentiment": news
    }


# --------------------------------------------------------------------------
# REPORT GENERATOR
# --------------------------------------------------------------------------
def report_generator_node(state: dict) -> dict:
    ticker = state["ticker"]
    predictions = state.get("predictions", "")
    news = state.get("news_sentiment", "")

    prompt = f"""
Write a clean Bloomberg-style markdown report.

PREDICTIONS:
{predictions}

NEWS:
{news}

End with: **Market Stance:** BULLISH/BEARISH/NEUTRAL | **Confidence:** High/Medium/Low
"""
    resp = llm.invoke([SystemMessage(content=prompt)])
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Extract stance
    upper = text.upper()
    stance = (
        "BULLISH" if "BULLISH" in upper else
        "BEARISH" if "BEARISH" in upper else
        "NEUTRAL"
    )

    # Confidence
    confidence = (
        "High" if "HIGH" in upper else
        "Low" if "LOW" in upper else
        "Medium"
    )

    return {
        "messages": [resp],
        "final_report": text,
        "recommendation": stance,
        "confidence": confidence
    }
