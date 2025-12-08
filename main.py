import os
import dspy
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv["GROQ_API_KEY"],
    max_tokens=512,
    temperature=0.2,
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = PersistentClient(path="./rag_chroma_store")

collection = chroma_client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "DSPy is a framework for building modular LLM pipelines.",
    "RAG stands for Retrieval Augmented Generation.",
    "Groq provides extremely fast inference for Llama-3 models.",
    "Chroma is a lightweight vector database used in RAG pipelines."
]

ids = [f"doc_{i}" for i in range(len(documents))]
embeddings = embedder.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)


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


class RAGAnswer(dspy.Signature):
        intent: str = dspy.InputField(desc="The purpose behind the user's query.") 
        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer: str = dspy.OutputField(
        desc="Answer using provided context only."
    )


class MyRAG(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retriever = retriever
        self.answerer = dspy.ChainOfThought(RAGAnswer)

    def forward(self, question):
        retrieved = self.retriever(question)
        context = "\n".join(retrieved.passages)

        result = self.answerer(
            intent="answer strictly using provided context only",
            question=question,
            context=context
        )
        
        return dspy.Prediction(
            answer=result.answer,
            context=context
        )
rag_pipeline = MyRAG(k=3)

query = "what is m4 in mac"
response = rag_pipeline(question=query)

print("\nQUESTION:")
print(query)

print("\nRETRIEVED CONTEXT:\n", response.context)

print("\nANSWER:\n", response.answer)
