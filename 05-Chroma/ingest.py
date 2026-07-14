import os
import chromadb
from dotenv import load_dotenv
from google import genai
from helper import embed_texts, extract_text, to_chunks

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chrom = chromadb.PersistentClient("./chroma.db")

sources = ["container_orchestr.pdf", "high_traffic_apis.pdf", "message_queues.pdf"]

documents = []
metadatas = []
ids = []

for source in sources:
    text = extract_text(source)
    chunks = to_chunks(text, chunk_size=1000)

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": source, "chunk_index": i})
        ids.append(f"{source}_{i}")

embeddings = embed_texts(client, documents)

try:
    chrom.delete_collection(name="Docs")
except (ValueError, chromadb.errors.NotFoundError):
    pass

collection = chrom.get_or_create_collection(
    name="Docs", metadata={"hnsw:space": "cosine"}
)

collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
