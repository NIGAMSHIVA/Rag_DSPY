# main.py

from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

from rag_engine import process_pdf, rag_pipeline

app = FastAPI()

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Upload PDF Route
# -------------------------------
@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    chunks = process_pdf(file_path)

    return {"status": "success", "chunks_added": chunks}


# -------------------------------
# Ask Query Route
# -------------------------------
@app.get("/query")
def ask_query(q: str):
    result = rag_pipeline(question=q)
    return {
        "question": q,
        "context": result.context,
        "answer": result.answer
    }


# Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
