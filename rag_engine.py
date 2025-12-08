# rag_engine.py
import os
import dspy
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ---------------------------
# Load environment + LLM
# ---------------------------
from dotenv import load_dotenv
load_dotenv()

print("DEBUG KEY =", os.getenv("GROQ_API_KEY"))



print("Loaded key:", os.getenv("GROQ_API_KEY"))

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=512,
    temperature=0.1,
)

# Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Chroma DB
chroma_client = PersistentClient(path="./db/rag_chroma_store")

collection = chroma_client.get_or_create_collection(
    name="pdf_docs",
    metadata={"hnsw:space": "cosine"}
)

# ---------------------------
# PDF Text Extraction
# ---------------------------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ---------------------------
# Chunk text
# ---------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# ---------------------------
# Upload PDF → embed → store
# ---------------------------
def process_pdf(file_path):
    raw_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(raw_text)

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    return len(chunks)


# ---------------------------
# DSPy Retrieval Module
# ---------------------------
class ChromaRM(dspy.Retrieve):
    def __init__(self, k=3):
        super().__init__(k=k)

    def forward(self, query):
        query_vec = embedder.encode([query]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=self.k
        )
        passages = results["documents"][0]
        return dspy.Prediction(passages=passages)


retriever = ChromaRM(k=3)
dspy.configure(lm=lm, rm=retriever)


# ---------------------------
# Signature + Answer Module
# ---------------------------
class RAGAnswer(dspy.Signature):
    intent: str = dspy.InputField()
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MyRAG(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retriever = retriever
        self.answerer = dspy.ChainOfThought(RAGAnswer)

    def forward(self, question):
        retrieved = self.retriever(question)
        context = "\n".join(retrieved.passages)

        result = self.answerer(
            intent="answer using only context",
            question=question,
            context=context
        )
        return dspy.Prediction(
            answer=result.answer,
            context=context
        )


rag_pipeline = MyRAG()
