import os
import time
import faiss
import numpy as np
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

para = """
Artificial Intelligence enables computers to perform tasks that normally require human intelligence. Machine learning allows AI systems to learn patterns from large amounts of data. Many companies use AI to automate customer support and business processes. Regular exercise improves physical fitness and strengthens the immune system. Walking or jogging for thirty minutes every day helps maintain a healthy heart. A balanced diet and regular physical activity are essential for a healthy lifestyle. Reading books expands knowledge and improves critical thinking skills. Atomic Habits by James Clear teaches the importance of building small, consistent habits. Many successful people read books daily to continue learning and growing. Climate change is causing rising global temperatures and extreme weather events. Deforestation and pollution contribute significantly to global warming. Using renewable energy can help reduce carbon emissions. AI is expected to play an even greater role in healthcare and education. Consistent exercise reduces stress and improves mental well-being. Reading for at least twenty minutes a day can significantly improve vocabulary and communication skills.
"""

clusters = [s.strip() for s in para.split(".") if s.strip()]

vectors = []
i = 0

for sentence in clusters:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=sentence,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    vectors.append(response.embeddings[0].values)
    print(f"cluster {i+1} is appended")
    i += 1
    time.sleep(5)

print("Loop is completed successfully")

query = "I love to do coding in silence"


response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
)

query_vector = response.embeddings[0].values

vectors = np.array(vectors, dtype="float32")
query_vector = np.array([query_vector], dtype="float32")

dimension = query_vector.shape[1]

centroid = faiss.IndexFlatL2(dimension)

nlist = 3

index = faiss.IndexIVFFlat(centroid, dimension, nlist, faiss.METRIC_L2)

index.train(vectors)  # Perform K-Mean Clustering

index.add(vectors)

index.nprobe = 1

distances, indexes = index.search(query_vector, 3)

print("Indexes")
print(indexes)

print("Distances")
print(distances)
