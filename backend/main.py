from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag_engine import process_pdf
from orchestrator import orchestrator_app

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# CORS (allow frontend access)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# File upload setup
# -------------------------------------------------
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str | None = None        # auto | rag | web | email
    rag_enabled: bool | None = None


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "RAG-DSPY"}

@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    chunks = process_pdf(file_path)

    return {
        "status": "success",
        "filename": pdf.filename,
        "chunks_added": chunks,
    }

@app.post("/query")
def ask_query(payload: QueryRequest):
    """
    Single endpoint:
    - auto → orchestrator decides intent
    - rag/web/email → forced intent
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


# -------------------------------------------------
# Uvicorn entrypoint (IMPORTANT for Render)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render injects PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
