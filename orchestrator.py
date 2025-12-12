# from __future__ import annotations

# from typing import TypedDict, Literal, Dict, Any
# import os

# import dspy
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, END

# import httpx
# from tavily import TavilyClient


# import re
# import smtplib
# from email.mime.text import MIMEText

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise RuntimeError("GROQ_API_KEY not found in environment")

# lm = dspy.LM(
#     model="groq/llama-3.1-8b-instant",
#     api_key=GROQ_API_KEY,
#     max_tokens=512,
#     temperature=0.1,
# )

# dspy.configure(lm=lm)

# from rag_engine import rag_pipeline  

# IntentType = Literal["rag", "web", "email"]

# class ConversationState(TypedDict, total=False):
#     query: str
#     intent: IntentType | None
#     answer: str | None
#     debug: Dict[str, Any]

# class ClassifyIntent(dspy.Signature):
#     """Decide whether a user query should go to RAG, Web Search, or Email agent.

#     - Use 'rag' when the query is about documents uploaded to the RAG system.
#     - Use 'web' when the query needs general internet knowledge or latest info.
#     - Use 'email' when the query asks to read, summarize, or analyze emails.
#     """

#     query = dspy.InputField(desc="The end user's natural-language question.")
#     intent = dspy.OutputField(
#         desc="One of: rag, web, email (lowercase single word)."
#     )


# class IntentRouter(dspy.Module):
#     """Tiny DSPy module that predicts the intent label."""

#     def __init__(self):
#         super().__init__()
#         self.predict = dspy.Predict(ClassifyIntent)

#     def forward(self, query: str) -> ClassifyIntent:
#         return self.predict(query=query)


# intent_router = IntentRouter()

# def rag_agent(state: ConversationState) -> ConversationState:
#     q = state["query"]
#     result = rag_pipeline(question=q)   
#     state["intent"] = "rag"
#     state["answer"] = result.answer
#     state.setdefault("debug", {})["rag_raw"] = getattr(result, "context", None)
#     return state

# def web_search_agent(state: ConversationState) -> ConversationState:
#     q = state["query"]
#     tavily_key = os.getenv("TAVILY_API_KEY")

#     if not tavily_key:
#         duckduckgo_url = "https://duckduckgo.com/?q=" + q.replace(" ", "+")

#         raw_output = lm(
#             f"User asked: {q}\n\n"
#             f"Provide a helpful general answer. "
#             f"You may optionally use this reference URL: {duckduckgo_url}."
#         )
#         if isinstance(raw_output, list):
#             raw_output = raw_output[0]

#         summary = str(raw_output).strip()

#         state["intent"] = "web"
#         state["answer"] = summary
#         state.setdefault("debug", {})["web_info"] = {
#             "engine": "duckduckgo_url_only"
#         }
#         return state
    
#     tavily_client = TavilyClient(api_key=tavily_key)
#     search_result = tavily_client.search(q, max_results=5)

#     raw_output = lm(
#         f"Summarize these web search results for the user's question.\n\n"
#         f"QUESTION:\n{q}\n\n"
#         f"WEB RESULTS:\n{search_result}"
#     )
#     if isinstance(raw_output, list):
#         raw_output = raw_output[0]

#     summary = str(raw_output).strip()

#     state["intent"] = "web"
#     state["answer"] = summary
#     state.setdefault("debug", {})["web_info"] = {
#         "engine": "tavily",
#         "raw": search_result
#     }

#     return state


#     q = state["query"]

#     email_host = os.getenv("EMAIL_HOST")
#     email_user = os.getenv("EMAIL_USER")
#     email_pass = os.getenv("EMAIL_PASSWORD")

#     def is_email_generation_request(query: str) -> bool:
#     keywords = [
#         "write email",
#         "draft email",
#         "resignation",
#     ]

#     return any(k in query.lower() for k in keywords)

# def email_agent(state: ConversationState) -> ConversationState:
#     q = state["query"].lower()

#     # 1. Extract email
#     email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
#     found_emails = re.findall(email_pattern, q)
#     receiver_email = found_emails[0] if found_emails else None

#     if not receiver_email:
#         state["intent"] = "email"
#         state["answer"] = "❌ No email address found in your query."
#         return state

#     # 2. Check if this is an email generation request
#     if is_email_generation_request(q):
#         purpose = q.replace(receiver_email, "").strip()

#         generated = generate_email_with_llm(
#             purpose=purpose,
#             receiver_email=receiver_email
#         )

#         state["intent"] = "email"
#         state["answer"] = (
#             f"📧 **Generated Email Preview**\n\n"
#             f"**To:** {receiver_email}\n"
#             f"**Subject:** {generated['subject']}\n\n"
#             f"{generated['body']}"
#         )

#         # Optional: store for sending later
#         state.setdefault("debug", {})["generated_email"] = generated
#         state["debug"]["email_mode"] = "generated_preview"

#         return state

#     # 3. Else fallback to manual send logic (your existing flow)
#     state["intent"] = "email"
#     state["answer"] = "❌ Please specify what email you want me to write."
#     return state


# def classify_intent_node(state: ConversationState) -> ConversationState:
#     q = state["query"]
#     prediction = intent_router(q)
#     intent_raw = (prediction.intent or "").strip().lower()

#     if intent_raw not in ("rag", "web", "email"):
#         # Safe default: use RAG``
#         intent_raw = "rag"

#     state["intent"] = intent_raw  # type: ignore
#     state.setdefault("debug", {})["router_raw"] = prediction.intent
#     return state


# def route_from_intent(state: ConversationState) -> str:
#     intent = state.get("intent", "rag")
#     if intent == "web":
#         return "web_agent"
#     if intent == "email":
#         return "email_agent"
#     return "rag_agent"  # default


# graph = StateGraph(ConversationState)

# graph.add_node("classify_intent", classify_intent_node)
# graph.add_node("rag_agent", rag_agent)
# graph.add_node("web_agent", web_search_agent)
# graph.add_node("email_agent", email_agent)

# graph.set_entry_point("classify_intent")

# graph.add_conditional_edges(
#     "classify_intent",
#     route_from_intent,
#     {
#         "rag_agent": "rag_agent",
#         "web_agent": "web_agent",
#         "email_agent": "email_agent",
#     },
# )

# graph.add_edge("rag_agent", END)
# graph.add_edge("web_agent", END)
# graph.add_edge("email_agent", END)

# orchestrator_app = graph.compile()

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

# ==================================================
# ENV + LLM
# ==================================================

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

# ==================================================
# TYPES
# ==================================================

IntentType = Literal["rag", "web", "email"]

class ConversationState(TypedDict, total=False):
    query: str
    intent: IntentType
    answer: str
    debug: Dict[str, Any]

# ==================================================
# INTENT ROUTER
# ==================================================

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

# ==================================================
# RAG AGENT
# ==================================================

def rag_agent(state: ConversationState) -> ConversationState:
    result = rag_pipeline(question=state["query"])
    state["intent"] = "rag"
    state["answer"] = result.answer
    return state

# ==================================================
# WEB AGENT
# ==================================================

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

# ==================================================
# EMAIL AGENT
# ==================================================

def email_agent(state: ConversationState) -> ConversationState:
    q = state["query"]

    # 1. Extract receiver email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    found = re.findall(email_pattern, q)
    receiver_email = found[0] if found else None

    if not receiver_email:
        state["intent"] = "email"
        state["answer"] = "❌ No email address found."
        return state

    # 2. Clean purpose text
    purpose = q.replace(receiver_email, "").strip()

    # 3. LLM PROMPT (STRICT FORMAT)
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

    # 4. Parse subject & body safely
    subject = "No Subject"
    body = raw_output

    if "SUBJECT:" in raw_output and "BODY:" in raw_output:
        subject = raw_output.split("SUBJECT:")[1].split("BODY:")[0].strip()
        subject = subject.replace("\\n", "\n")
        body = raw_output.split("BODY:")[1].strip()

        body = body.replace("\\n", "\n")


    # 5. SMTP CONFIG
    email_host = os.getenv("EMAIL_HOST")
    email_port = int(os.getenv("EMAIL_PORT", "587"))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASSWORD")

    if not all([email_host, email_user, email_pass]):
        state["intent"] = "email"
        state["answer"] = "❌ Email server credentials missing."
        return state

    # 6. SEND EMAIL
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
        state["answer"] = f"✅ Email sent successfully to {receiver_email}."

        return state

    except Exception as e:
        state["intent"] = "email"
        state["answer"] = f"❌ Failed to send email: {str(e)}"
        return state

# ==================================================
# ROUTING
# ==================================================

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

# ==================================================
# LANGGRAPH
# ==================================================

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
