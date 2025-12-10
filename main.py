# main.py

from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

from rag_engine import process_pdf  # you already have this
from orchestrator import orchestrator_app  # NEW: LangGraph app

app = FastAPI()

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    chunks = process_pdf(file_path)

    return {"status": "success", "chunks_added": chunks}


@app.get("/query")
def ask_query(q: str):
    """Single endpoint that goes through LangGraph multi-agent orchestrator."""
    initial_state = {
        "query": q,
        "intent": None,
        "answer": None,
        "debug": {},
    }

    final_state = orchestrator_app.invoke(initial_state)

    return {
        "question": q,
        "intent": final_state.get("intent"),
        "answer": final_state.get("answer"),
        "debug": final_state.get("debug"),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
