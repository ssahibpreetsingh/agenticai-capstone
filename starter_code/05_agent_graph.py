"""
05_agent_graph.py — Module 5: Agentic AI (Tools + LangGraph + Human-in-the-Loop)

GOAL: Wrap your grounded RAG pipeline as a TOOL inside a LangGraph agent, add a second
utility tool (EMI calculator for banking / date utility for healthcare), and add a
human-in-the-loop approval gate before any "risky" action (e.g. confirming a loan
action, or finalising a medication-related recommendation).

This mirrors your Module 5 Option A (single ReAct agent) + Option C (human-in-the-loop)
labs, minus Azure-specific tracing.

Time budget: ~25 minutes (2nd tool + HITL gate; if short on time, at minimum get the
RAG tool + one utility tool working, and stub the approval gate with a simple
input()-based confirmation — that still counts as HITL for this capstone)
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from typing import TypedDict, Annotated
from importlib import import_module
import operator

from config import SCENARIO, chat

generate_module = import_module("04_generate_grounded")


# ---------------------------------------------------------------------------
# TOOL 1: RAG lookup (wraps your Stage 4 grounded generation pipeline)
# ---------------------------------------------------------------------------

def rag_lookup_tool(query: str) -> str:
    """Answers a policy/FAQ question using the grounded RAG pipeline."""
    result = generate_module.answer_question(query)
    return result["answer"]


# ---------------------------------------------------------------------------
# TOOL 2: scenario-specific utility tool
# ---------------------------------------------------------------------------

def emi_calculator_tool(principal: float, annual_rate_percent: float, tenure_months: int) -> str:
    """Calculates EMI using the reducing balance formula (Module 5.3 example: banking)."""
    r = annual_rate_percent / 12 / 100
    n = tenure_months
    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return f"Estimated EMI: INR {emi:,.2f} per month for {n} months at {annual_rate_percent}% p.a."


def date_utility_tool(reference_date: str, days_to_add: int) -> str:
    """
    Simple date utility (Module 5.3 example: healthcare) — e.g. "when is my follow-up
    due" or "is my application window still open" type questions.

    TODO: expand this if you want a richer tool — e.g. actually check today's date
    against an application/appointment window from the policy documents.
    """
    from datetime import datetime, timedelta
    ref = datetime.strptime(reference_date, "%Y-%m-%d")
    result_date = ref + timedelta(days=days_to_add)
    return f"{days_to_add} days from {reference_date} is {result_date.strftime('%Y-%m-%d')}."


SCENARIO_TOOLS = {
    "banking": {"rag_lookup": rag_lookup_tool, "calculate_emi": emi_calculator_tool},
    "healthcare": {"rag_lookup": rag_lookup_tool, "date_utility": date_utility_tool},
}


# ---------------------------------------------------------------------------
# Human-in-the-loop approval gate (Module 5.7)
# ---------------------------------------------------------------------------

RISKY_KEYWORDS = {
    "banking": ["approve", "disburse", "transfer", "close account", "foreclose"],
    "healthcare": ["prescribe", "adjust dose", "recommend medication", "change dosage"],
}


def requires_human_approval(user_query: str) -> bool:
    """Simple keyword-based risk check — TODO: make this smarter if you have time
    (e.g. ask the LLM to classify risk level instead of keyword matching)."""
    keywords = RISKY_KEYWORDS.get(SCENARIO, [])
    return any(k in user_query.lower() for k in keywords)


def human_approval_gate(proposed_action: str) -> bool:
    """
    Pauses execution and asks a human to approve/reject before a risky action proceeds.
    """
    # # In this capstone we simulate this via console input() — in production this would be
    # a real UI approval step or a LangGraph `interrupt()` (see TODO below).

    # TODO (stretch): replace this console-based gate with a real LangGraph interrupt
    # using `langgraph.types.interrupt` so the graph actually pauses/resumes state,
    # rather than blocking synchronously on input(). See LangGraph docs on
    # human-in-the-loop for the exact pattern from your Module 5.7 lab.
    print(f"\n[HUMAN APPROVAL REQUIRED]\nProposed action: {proposed_action}")
    response = input("Approve? (y/n): ").strip().lower()
    return response == "y"


# ---------------------------------------------------------------------------
# Minimal agent loop
#
# TODO: This is a simplified, hand-rolled ReAct-style loop so the capstone runs
# without requiring you to debug full LangGraph StateGraph wiring under time
# pressure. For full credit, port this into an actual LangGraph StateGraph with
# nodes/edges (as in your Module 5.6 lab) — see the commented skeleton below.
# ---------------------------------------------------------------------------

def run_agent(user_query: str) -> str:
    tools = SCENARIO_TOOLS.get(SCENARIO, SCENARIO_TOOLS["healthcare"])

    # Step 1: decide if this needs the RAG tool (almost always yes for informational Qs)
    print(f"\n[AGENT] Query: {user_query}")
    print("[AGENT] Step 1: calling rag_lookup tool...")
    rag_answer = tools["rag_lookup"](user_query)
    print(f"[AGENT] RAG tool result: {rag_answer}")

    # Step 2: human-in-the-loop gate for risky actions
    if requires_human_approval(user_query):
        approved = human_approval_gate(f"Respond to risky request: '{user_query}'")
        if not approved:
            return "Action not approved by human reviewer. No further action taken."

    return rag_answer


# ---------------------------------------------------------------------------
# TODO (full-credit path): real LangGraph StateGraph skeleton
# ---------------------------------------------------------------------------

from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    rag_result: str
    needs_approval: bool
    approved: bool
    final_answer: str

def rag_node(state: AgentState) -> AgentState:
    tools = SCENARIO_TOOLS.get(SCENARIO, SCENARIO_TOOLS["healthcare"])
    state["rag_result"] = tools["rag_lookup"](state["query"])
    return state

def approval_check_node(state: AgentState) -> AgentState:
    state["needs_approval"] = requires_human_approval(state["query"])
    return state

def approval_gate_node(state: AgentState) -> AgentState:
    state["approved"] = human_approval_gate(state["query"])
    return state

def finalize_node(state: AgentState) -> AgentState:
    if state.get("needs_approval") and not state.get("approved", True):
        state["final_answer"] = "Action not approved."
    else:
        state["final_answer"] = state["rag_result"]
    return state



def main():
    demo_queries = {
        "banking": [
            "Can I postpone my EMI payment?",
            "Please approve a transfer to close my account and disburse the balance.",
        ],
        "healthcare": [
            "How often should my HbA1c be checked?",
            "Please adjust my dose based on my last lab result.",
        ],
    }

    tools = SCENARIO_TOOLS.get(SCENARIO, SCENARIO_TOOLS["healthcare"])
    queries = demo_queries.get(SCENARIO, demo_queries["banking"])
    queries.append("Approve my loan of 2000000")
    # for q in queries:
    #     answer = run_agent(q)
    #     print(f"\n[FINAL ANSWER] {answer}\n{'-'*70}")
    graph = StateGraph(AgentState)
    graph.add_node("rag", rag_node)
    graph.add_node("check_approval", approval_check_node)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("rag")
    graph.add_edge("rag", "check_approval")
    graph.add_conditional_edges(
        "check_approval",
        lambda s: "approval_gate" if s["needs_approval"] else "finalize",
    )
    graph.add_edge("approval_gate", "finalize")
    graph.add_edge("finalize", END)

    app = graph.compile()

    for q in queries[:]:
        
        # answer = run_agent(q)
        result = app.invoke({"query": q})
        print("result: ",result)
   
    # print(result)

if __name__ == "__main__":
    main()
