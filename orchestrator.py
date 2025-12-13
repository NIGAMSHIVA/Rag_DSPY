from __future__ import annotations

from typing import TypedDict, Literal, Dict, Any
import os
import re
import smtplib

import dspy
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from email.mime.text import MIMEText

from rag_engine import rag_pipeline



load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=700,
)

dspy.configure(lm=lm)


IntentType = Literal["rag", "web", "email"]

class ConversationState(TypedDict, total=False):
    query: str
    intent: IntentType
    answer: str
    debug: Dict[str, Any]


class ClassifyIntent(dspy.Signature):
    query = dspy.InputField()
    intent = dspy.OutputField(desc="rag | web | email")

class IntentRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ClassifyIntent)

    def forward(self, query: str):
        return self.predict(query=query)

intent_router = IntentRouter()


def rag_agent(state: ConversationState) -> ConversationState:
    result = rag_pipeline(question=state["query"])
    state["intent"] = "rag"
    state["answer"] = result.answer
    return state


def web_search_agent(state: ConversationState) -> ConversationState:
    q = state["query"]
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        out = lm(f"Answer clearly:\n{q}")
        state["intent"] = "web"
        state["answer"] = str(out).strip()
        return state

    client = TavilyClient(api_key=tavily_key)
    results = client.search(q, max_results=5)

    out = lm(f"Summarize these web results clearly:\n{results}")
    state["intent"] = "web"
    state["answer"] = str(out).strip()
    return state

def email_agent(state: ConversationState) -> ConversationState:
    q = state["query"]

    # 1. Extract receiver email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    found = re.findall(email_pattern, q)
    receiver_email = found[0] if found else None

    if not receiver_email:
        state["intent"] = "email"
        state["answer"] = " No email address found."
        return state

    purpose = q.replace(receiver_email, "").strip()

    prompt = f"""
You are a professional email writer.

Return ONLY in this format:

SUBJECT:
<subject here>

BODY:
<body here>

Context: {purpose}

Rules:
- No quotes
- No brackets
- Professional tone
"""

    raw_output = str(lm(prompt)).strip()

    subject = "No Subject"
    body = raw_output

    if "SUBJECT:" in raw_output and "BODY:" in raw_output:
        subject = raw_output.split("SUBJECT:")[1].split("BODY:")[0].strip()
        subject = subject.replace("\\n", "\n")
        body = raw_output.split("BODY:")[1].strip()

        body = body.replace("\\n", "\n")


    email_host = os.getenv("EMAIL_HOST")
    email_port = int(os.getenv("EMAIL_PORT", "587"))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASSWORD")

    if not all([email_host, email_user, email_pass]):
        state["intent"] = "email"
        state["answer"] = " Email server credentials missing."
        return state

    try:
        msg = MIMEText(body)
        msg["From"] = email_user
        msg["To"] = receiver_email
        msg["Subject"] = subject

        server = smtplib.SMTP(email_host, email_port)
        server.starttls()
        server.login(email_user, email_pass)
        server.sendmail(email_user, receiver_email, msg.as_string())
        server.quit()

        state["intent"] = "email"
        state["answer"] = f" Email sent successfully to {receiver_email}."

        return state

    except Exception as e:
        state["intent"] = "email"
        state["answer"] = f" Failed to send email: {str(e)}"
        return state


def classify_intent_node(state: ConversationState) -> ConversationState:
    pred = intent_router(state["query"])
    state["intent"] = (pred.intent or "rag").lower()
    return state

def route_from_intent(state: ConversationState) -> str:
    if state["intent"] == "web":
        return "web_agent"
    if state["intent"] == "email":
        return "email_agent"
    return "rag_agent"


graph = StateGraph(ConversationState)

graph.add_node("classify_intent", classify_intent_node)
graph.add_node("rag_agent", rag_agent)
graph.add_node("web_agent", web_search_agent)
graph.add_node("email_agent", email_agent)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_from_intent,
    {
        "rag_agent": "rag_agent",
        "web_agent": "web_agent",
        "email_agent": "email_agent",
    },
)

graph.add_edge("rag_agent", END)
graph.add_edge("web_agent", END)
graph.add_edge("email_agent", END)

orchestrator_app = graph.compile()
