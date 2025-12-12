"""
LangGraph assembly + analyze_stock wrapper.
No optional Finnhub settings.
"""
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from src.agents.nodes import (
    performance_analyst_node,
    market_expert_node,
    report_generator_node
)


class AgentState(MessagesState):
    ticker: str
    predictions: str
    news_sentiment: str
    final_report: str
    recommendation: str
    confidence: str


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("perf", performance_analyst_node)
    g.add_node("news", market_expert_node)
    g.add_node("report", report_generator_node)

    g.set_entry_point("perf")
    g.add_edge("perf", "news")
    g.add_edge("news", "report")
    g.add_edge("report", END)

    return g.compile(checkpointer=MemorySaver())


def analyze_stock(ticker: str, thread_id: str = None):
    # FIRST attempt prediction directly
    from src.agents.tools import fetch_prediction_data

    # Use the new fetcher
    raw_data = fetch_prediction_data(ticker)

    # ---- FIX: MODEL TRAINING CASE ----
    if raw_data == "__MODEL_TRAINING__":
        return {
            "status": "training",
            "detail": f"Model for {ticker} is being trained. Retry after a few seconds.",
            "ticker": ticker.upper()
        }
    
    # Check for error string
    if isinstance(raw_data, str):
        # This means an error occurred (e.g. 500 or 404 text)
        return {
            "status": "error",
            "detail": raw_data,
            "ticker": ticker.upper()
        }

    # Format for Agent (String)
    # Re-implement the formatting logic briefly or call the tool? 
    # Calling the tool would re-fetch. Let's format manually to save a call.
    try:
        forecast = (
            raw_data.get("result", {})
            .get("predictions", {})
            .get("full_forecast", [])
        )
        if not forecast:
            pred_str = f"No predictions available for {ticker}"
        else:
            lines = [f"7-Day Price Forecast for {ticker}:"]
            for row in forecast[:7]:
                price = float(row.get("close", 0))
                lines.append(f"  {row['date']}: ${price:.2f}")
            pred_str = "\n".join(lines)
    except Exception as e:
        pred_str = f"Prediction parsing failed: {e}"

    # Continue only if predictions available
    graph = build_graph()

    state = {
        "ticker": ticker.upper(),
        "messages": [HumanMessage(content=f"Start analysis {ticker}")],
        "predictions": pred_str 
    }

    config = {"configurable": {"thread_id": thread_id or "1"}}
    
    # Invoke the graph
    result = graph.invoke(state, config=config)
    
    # Inject RAW data for frontend (so it has history)
    # We put it in the 'predictions' key of the DICT result (not State) or a new key
    # The result is a dict of the final state.
    # We can add 'predictions_data' to it.
    if isinstance(raw_data, dict):
        # Extract the useful part
        result["predictions"] = raw_data.get("result", {}).get("predictions", {})
    
    return result
