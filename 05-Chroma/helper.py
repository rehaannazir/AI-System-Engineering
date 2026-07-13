import time

import fitz
from google.genai import types


def extract_text(path):
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def to_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def embed_texts(client, texts, model="gemini-embedding-001", delay=1):
    embeddings = []

    for text in texts:
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings.append(response.embeddings[0].values)
        time.sleep(delay)

    return embeddings
