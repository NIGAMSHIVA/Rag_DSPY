from __future__ import annotations

from typing import TypedDict, Literal, Dict, Any
import os

import dspy
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

import httpx
from tavily import TavilyClient

import imaplib
import email
from email.header import decode_header
import re
import smtplib
from email.mime.text import MIMEText

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    max_tokens=512,
    temperature=0.1,
)

dspy.configure(lm=lm)

from rag_engine import rag_pipeline  

IntentType = Literal["rag", "web", "email"]

class ConversationState(TypedDict, total=False):
    query: str
    intent: IntentType | None
    answer: str | None
    debug: Dict[str, Any]

class ClassifyIntent(dspy.Signature):
    """Decide whether a user query should go to RAG, Web Search, or Email agent.

    - Use 'rag' when the query is about documents uploaded to the RAG system.
    - Use 'web' when the query needs general internet knowledge or latest info.
    - Use 'email' when the query asks to read, summarize, or analyze emails.
    """

    query = dspy.InputField(desc="The end user's natural-language question.")
    intent = dspy.OutputField(
        desc="One of: rag, web, email (lowercase single word)."
    )


class IntentRouter(dspy.Module):
    """Tiny DSPy module that predicts the intent label."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ClassifyIntent)

    def forward(self, query: str) -> ClassifyIntent:
        return self.predict(query=query)


intent_router = IntentRouter()

def rag_agent(state: ConversationState) -> ConversationState:
    q = state["query"]
    result = rag_pipeline(question=q)   
    state["intent"] = "rag"
    state["answer"] = result.answer
    state.setdefault("debug", {})["rag_raw"] = getattr(result, "context", None)
    return state

def web_search_agent(state: ConversationState) -> ConversationState:
    q = state["query"]
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        duckduckgo_url = "https://duckduckgo.com/?q=" + q.replace(" ", "+")

        raw_output = lm(
            f"User asked: {q}\n\n"
            f"Provide a helpful general answer. "
            f"You may optionally use this reference URL: {duckduckgo_url}."
        )
        if isinstance(raw_output, list):
            raw_output = raw_output[0]

        summary = str(raw_output).strip()

        state["intent"] = "web"
        state["answer"] = summary
        state.setdefault("debug", {})["web_info"] = {
            "engine": "duckduckgo_url_only"
        }
        return state
    
    tavily_client = TavilyClient(api_key=tavily_key)
    search_result = tavily_client.search(q, max_results=5)

    raw_output = lm(
        f"Summarize these web search results for the user's question.\n\n"
        f"QUESTION:\n{q}\n\n"
        f"WEB RESULTS:\n{search_result}"
    )
    if isinstance(raw_output, list):
        raw_output = raw_output[0]

    summary = str(raw_output).strip()

    state["intent"] = "web"
    state["answer"] = summary
    state.setdefault("debug", {})["web_info"] = {
        "engine": "tavily",
        "raw": search_result
    }

    return state


    q = state["query"]

    email_host = os.getenv("EMAIL_HOST")
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASSWORD")

    # If IMAP credentials missing -> fallback to mock mode
    if not (email_host and email_user and email_pass):
        mock_emails = [
            "Meeting with product team tomorrow at 3 PM.",
            "HR: Please submit your documents.",
            "Reminder: Weekly report is due Friday."
        ]

        raw_out = lm(
            f"User query: {q}\n\n"
            f"Emails:\n{mock_emails}\n\n"
            f"Give the best possible answer based on these emails."
        )
        if isinstance(raw_out, list):
            raw_out = raw_out[0]

        state["intent"] = "email"
        state["answer"] = str(raw_out).strip()
        state.setdefault("debug", {})["email_info"] = {"mode": "mock"}
        return state

    # REAL GMAIL IMAP MODE
    try:
        mail = imaplib.IMAP4_SSL(email_host)
        mail.login(email_user, email_pass)
        mail.select("inbox")

        # Get last 10 emails
        _, data = mail.search(None, "ALL")
        mail_ids = data[0].split()
        latest_ids = mail_ids[-10:]   # last 10 emails

        extracted_emails = []

        for num in latest_ids:
            _, msg_data = mail.fetch(num, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Decode subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            # Extract body
            body_text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body_text = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        break
            else:
                body_text = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

            extracted_emails.append({
                "subject": subject,
                "body": body_text[:500]  # limit body length
            })

        # Summarize using DSPy LM
        raw_out = lm(
            f"User query: {q}\n\n"
            f"Here are the last {len(extracted_emails)} emails:\n{extracted_emails}\n\n"
            f"Provide the most useful summary or answer."
        )
        if isinstance(raw_out, list):
            raw_out = raw_out[0]

        answer = str(raw_out).strip()

        state["intent"] = "email"
        state["answer"] = answer
        state.setdefault("debug", {})["email_info"] = {
            "mode": "imap",
            "emails_count": len(extracted_emails)
        }
        return state

    except Exception as e:
        state["intent"] = "email"
        state["answer"] = f"Email agent error: {str(e)}"
        state.setdefault("debug", {})["email_info"] = {"mode": "error"}
        return state
def email_agent(state: ConversationState) -> ConversationState:
    q = state["query"].lower()

   
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    found_emails = re.findall(email_pattern, q)
    receiver_email = found_emails[0] if found_emails else None

    
    message_body = None

    if "saying" in q:
        message_body = q.split("saying", 1)[1].strip()
    elif "message" in q:
        message_body = q.split("message", 1)[1].strip()
    elif "body" in q:
        message_body = q.split("body", 1)[1].strip()

  
    if not receiver_email:
        state["intent"] = "email_send"
        state["answer"] = " No email address found in your query."
        return state

    if not message_body:
        state["intent"] = "email_send"
        state["answer"] = " No message/body found in your query."
        return state

  
    email_host = os.getenv("EMAIL_HOST")
    email_port = int(os.getenv("EMAIL_PORT", "587"))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASSWORD")

    if not (email_host and email_user and email_pass):
        state["intent"] = "email_send"
        state["answer"] = " Email credentials missing in server configuration."
        return state


    try:
        msg = MIMEText(message_body)
        msg["Subject"] = "Message from DSPy Email Agent"
        msg["From"] = email_user
        msg["To"] = receiver_email

        server = smtplib.SMTP(email_host, email_port)
        server.starttls()
        server.login(email_user, email_pass)
        server.sendmail(email_user, receiver_email, msg.as_string())
        server.quit()

        state["intent"] = "email_send"
        state["answer"] = f" Email sent successfully to {receiver_email}!"
        return state

    except Exception as e:
        state["intent"] = "email_send"
        state["answer"] = f"Failed to send email: {str(e)}"
        return state


def classify_intent_node(state: ConversationState) -> ConversationState:
    q = state["query"]
    prediction = intent_router(q)
    intent_raw = (prediction.intent or "").strip().lower()

    if intent_raw not in ("rag", "web", "email"):
        # Safe default: use RAG``
        intent_raw = "rag"

    state["intent"] = intent_raw  # type: ignore
    state.setdefault("debug", {})["router_raw"] = prediction.intent
    return state


def route_from_intent(state: ConversationState) -> str:
    intent = state.get("intent", "rag")
    if intent == "web":
        return "web_agent"
    if intent == "email":
        return "email_agent"
    return "rag_agent"  # default


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
