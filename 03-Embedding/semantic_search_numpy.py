import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

paragrapgh = """
Artificial intelligence is transforming healthcare.Machine learning models require quality data.Deep learning uses neural networks.Large language models generate human-like text.AI agents automate repetitive business tasks.Python is a popular programming language.FastAPI helps build high-performance APIs.Git is used for version control.JavaScript powers interactive websites.Software testing improves application reliability.Regular exercise improves physical health.Drinking enough water keeps the body hydrated.A balanced diet supports overall wellness.Good sleep improves mental health.Walking every day reduces stress.Investing early helps build long-term wealth.Risk management is essential in trading.Diversification reduces investment risk.Inflation decreases purchasing power.Stock markets react to economic news.Dogs are loyal companions.Cats are independent animals.Dolphins are intelligent marine mammals.Elephants have excellent memory.Birds migrate during different seasons.Students learn better through consistent practice.Reading books improves vocabulary.Mathematics develops logical thinking.Online education is becoming more popular.Teachers inspire lifelong learning.
"""

sentences = paragrapgh.split(".")
vector_db = []
i = 0

for sentence in sentences:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence,
        config=types.EmbedContentConfigDict(task_type="SEMANTIC_SIMILARITY"),
    )

    vector_db.append(response.embeddings)
    print(f"Vector {i+1} is appended")
    time.sleep(5)


vector_matrix = np.array(vector_db)
