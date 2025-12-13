from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag_engine import process_pdf
from orchestrator import orchestrator_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    mode: str | None = None        
    rag_enabled: bool | None = None  

@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    chunks = process_pdf(file_path)

    return {"status": "success", "chunks_added": chunks}

@app.post("/query")
def ask_query(payload: QueryRequest):
    """
    Single endpoint for all:
    - auto: orchestrator decides intent (web/email/rag/etc)
    - rag/web/email: user forces the intent
    """
    initial_state = {
        "query": payload.query,
        "intent": payload.mode if payload.mode and payload.mode != "auto" else None,
        "answer": None,
        "debug": {
            "rag_enabled": payload.rag_enabled
        },
    }

    final_state = orchestrator_app.invoke(initial_state)

    return {
        "question": payload.query,
        "intent": final_state.get("intent"),
        "answer": final_state.get("answer"),
        "debug": final_state.get("debug"),
    }
