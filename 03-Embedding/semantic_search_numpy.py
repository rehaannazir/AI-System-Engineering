import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

para = """
Artificial intelligence is transforming healthcare.Machine learning models require quality data.Deep learning uses neural networks.Large language models generate human-like text.AI agents automate repetitive business tasks.Neural networks are inspired by the human brain.Reinforcement learning teaches agents through rewards.Computer vision helps machines interpret images.Natural language processing enables text understanding.Python is a popular programming language.FastAPI helps build high-performance APIs.Git is used for version control.JavaScript powers interactive websites.Software testing improves application reliability.Docker simplifies application deployment.Cloud computing offers scalable infrastructure.Databases store and organize structured data.Regular exercise improves physical health.Drinking enough water keeps the body hydrated.A balanced diet supports overall wellness.Good sleep improves mental health.Walking every day reduces stress.Meditation helps calm an anxious mind.Routine checkups catch illnesses early.Stretching prevents muscle injuries.Investing early helps build long-term wealth.Risk management is essential in trading.Diversification reduces investment risk.Inflation decreases purchasing power.Stock markets react to economic news.Budgeting helps control monthly expenses.Compound interest grows savings over time.Emergency funds provide financial security.Dogs are loyal companions.Cats are independent animals.Dolphins are intelligent marine mammals.Elephants have excellent memory.Birds migrate during different seasons.Lions live and hunt in prides.Bees play a vital role in pollination.Wolves communicate through howling.Students learn better through consistent practice.Reading books improves vocabulary.Mathematics develops logical thinking.Online education is becoming more popular.Teachers inspire lifelong learning.Group projects build teamwork skills.Curiosity drives effective learning.Exams measure a student's understanding.Libraries provide access to vast knowledge.
"""

sentences = [s.strip() for s in para.split(".") if s.strip()]  # Chunking

# Topic label for each sentence, in the same order as `para`
topics = ["AI", "Technology", "Health", "Finance", "Animals", "Education"]
topic_counts = [9, 8, 8, 8, 8, 9]  # how many sentences belong to each topic, sums to 50
labels = [topic for topic, count in zip(topics, topic_counts) for _ in range(count)]
vector_db = []
i = 0

for sentence in sentences:

    response = client.models.embed_content(  # Embedding
        model="gemini-embedding-001",
        contents=sentence,
        config=types.EmbedContentConfigDict(task_type="SEMANTIC_SIMILARITY"),
    )

    vector_db.append(response.embeddings[0].values)
    print(f"Vector {i+1} is appended")
    time.sleep(5)
    i += 1

print("*********** >Completed< ***************\n")


vector_matrix = np.array(vector_db)

query = "Just digging deeper in RAG"

query_embed = client.models.embed_content(  # Query Embedding
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
)

query_vector = query_embed.embeddings[0].values

# Cosine Similarity

cosines = []

for i, embedding in enumerate(vector_matrix):

    result = (np.dot(query_vector, embedding)) / (
        np.linalg.norm(query_vector) * np.linalg.norm(embedding)
    )

    cosines.append((i, result))

# Top-K Search

cosines.sort(
    key=lambda x: x[1], reverse=True
)  # To sort on behalf of cosine values not index so x is list value and x[1] gives cosine

top_3_scores = cosines[:3]

# Retrive actual sentence

for index, score in top_3_scores:

    print(f"{sentences[index]} : {score}")

# PCA

pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(vector_matrix)


plt.figure(figsize=(10, 6))

labels_array = np.array(labels)

for topic in topics:
    # Pick only the points that belong to this topic
    topic_points = pca_vectors[labels_array == topic]
    plt.scatter(topic_points[:, 0], topic_points[:, 1], label=topic)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Embeddings Spread")
plt.legend(title="Topic")
plt.show()
