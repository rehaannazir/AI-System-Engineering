import os
import time
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from helper import extract_text, to_chunks, embed_texts

# Getting key from env
load_dotenv()

# Setting Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Extracting text from PDF
pdf_text = extract_text("container_orchestr.pdf")

# Chunking text
chunks = to_chunks(pdf_text, chunk_size=500)

# Embedding Text
vectors = embed_texts(client=client, texts=chunks)

# Creating Chromadb
chroma = chromadb.PersistentClient("./chroma.db")

# Creating Collection
collection = chroma.get_or_create_collection(
    name="Docs", metadata={"hnsw:space": "cosine"}
)

# Creating chunk features
ids = []
metadatas = []

for i, vector in enumerate(vectors):

    ids.append(f"doc_1_{i}")
    metadatas.append({"source": "container_orchestr.pdf", "author": "Rehan Nazir"})

# Ingesting in database
collection.add(
    documents=chunks,
    embeddings=vectors,
    ids=ids,
    metadatas=metadatas,
)

# Query Forming
query = "What autoscaling metric does RabbitMQ use by default?"

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
)

query_vec = response.embeddings[0].values

# Retrieving
result = collection.query(query_embeddings=[query_vec], n_results=3)

context = result["documents"][0]
source = result["metadatas"][0]

# Augmenting (Context Stuffing)
prompt = f"""
Answer the following query from the context.

query = {query}

Answer only from 
context = {context}

Also mention it if answer is from context
source = {source} 

Otherwise say "I don't know"

e.g.

query = "Who is founder of pakistan?"

context = ["the car is made by china", "flying in the sky and move around", "Quaid the founder of pakistan"]
source = ["pdf_1, #2", "pdf_2, #18", "pdf_3, #14"]

Your Response : According to pdf_3 id no #14, the founder of Pakistan is Quaid e Azam
"""

result = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="You are mentor responsible for answering queries of people. You will be provided by three things query(The question by user), context(You answer restricted to this context) and source(The metadata of context you can mention it in answer). Answer the query from the context only. Your answers should be restricted to context and mention source. If answer is not present in context just say 'I don't know sorry!'"
    ),
)

# Generating
full_text = ""

for chunk in result:
    if chunk.text:
        full_text += chunk.text
        for char in chunk.text:
            print(char, end="", flush=True)
            time.sleep(0.02)
