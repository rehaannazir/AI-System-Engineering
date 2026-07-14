import os

import chromadb
from dotenv import load_dotenv
from google import genai

from helper import embed_texts

load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chrom = chromadb.PersistentClient("./chroma.db")

collection = chrom.get_or_create_collection(
    name="Docs", metadata={"hnsw:space": "cosine"}
)

query = "How do you configure a visibility timeout in Redis?"
query_embed = embed_texts(client, [query])[0]

results = collection.query(query_embeddings=[query_embed], n_results=3)

retrieved_chunks = results["documents"][0]
metas = results["metadatas"][0]
dists = results["distances"][0]

for chunk, meta, dist in zip(retrieved_chunks, metas, dists):

    print(f"[{meta['source']} #{meta['chunk_index']}] (distance={dist:.4f})")
    print(chunk)
    print("---")
