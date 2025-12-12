# # main.py

# from fastapi import FastAPI, UploadFile, File
# import uvicorn
# import os

# from rag_engine import process_pdf  # you already have this
# from orchestrator import orchestrator_app  # NEW: LangGraph app
# from fastapi.middleware.cors import CORSMiddleware

# class QueryRequest(BaseModel):
#     query: str



# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",  # Next.js frontend
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_FOLDER = "./uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# @app.post("/upload-pdf")
# async def upload_pdf(pdf: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

#     with open(file_path, "wb") as f:
#         f.write(await pdf.read())

#     chunks = process_pdf(file_path)

#     return {"status": "success", "chunks_added": chunks}


# @app.get("/query")
# def ask_query(payload: QueryRequest):
#     """Single endpoint that goes through LangGraph multi-agent orchestrator."""
#     initial_state = {
#         "query": payload.query,
#         "intent": None,
#         "answer": None,
#         "debug": {},
#     }

#     final_state = orchestrator_app.invoke(initial_state)

#     return {
#         "question":  payload.query,
#         "intent": final_state.get("intent"),
#         "answer": final_state.get("answer"),
#         "debug": final_state.get("debug"),
#     }


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from rag_engine import process_pdf
from orchestrator import orchestrator_app

# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    query: str


app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Upload PDF
# -----------------------------
@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    chunks = process_pdf(file_path)
    return {"status": "success", "chunks_added": chunks}

# -----------------------------
# QUERY (POST ✅)
# -----------------------------
@app.post("/query")
def ask_query(payload: QueryRequest):
    initial_state = {
        "query": payload.query,
        "intent": None,
        "answer": None,
        "debug": {},
    }

    final_state = orchestrator_app.invoke(initial_state)

    return {
        "question": payload.query,
        "intent": final_state.get("intent"),
        "answer": final_state.get("answer"),
        "debug": final_state.get("debug"),
    }

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
