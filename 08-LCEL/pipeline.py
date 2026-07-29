import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "06-Naive RAG"))
from helper import extract_text, to_chunks  # noqa: E402

# Loading values from env
load_dotenv()

# Making LLM model runnable
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Making Embedding Model
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Loading text from DOC (path resolved relative to the repo root, not the CWD)
PDF_PATH = REPO_ROOT / "06-Naive RAG" / "container_orchestr.pdf"
doc_text = extract_text(str(PDF_PATH))

# Generating chunks from text
doc_chunks = to_chunks(doc_text, chunk_size=500, overlap=100)

if not doc_chunks:
    raise ValueError(f"No extractable text found in {PDF_PATH}")

# Getting user  query
query = input("\n Enter your query: ")

# Generating Vector Store
vector_store = FAISS.from_texts(texts=doc_chunks, embedding=embedding)


# As above return doc part but we want text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Making Retriever
retriever = vector_store.as_retriever()

# Making prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the following query from the context.

query = {query}

Answer only from 
context = {context}

Otherwise say "I don't know"

e.g.

query = "Who is founder of pakistan?"

context = ["the car is made by china", "flying in the sky and move around", "Quaid the founder of pakistan"]

Your Response : The founder of Pakistan is Quaid e Azam
""")

# Preparing remaining runnables
pass_through = RunnablePassthrough()
parser = StrOutputParser()
doc_context = RunnableLambda(format_docs)


# Time ro make CHAIN (Pipeline; the spine of AI application)
chain = (
    RunnableParallel(query=pass_through, context=retriever | doc_context)
    | prompt_template
    | llm
    | parser
)

# Simple response
# answer = chain.invoke(query)
# print(answer)

# Stream response
# full_text = ""

# for chunk in chain.stream(query):
#     if chunk:
#         full_text += chunk
#         for char in chunk:
#             print(char, end="", flush=True)
#             time.sleep(0.02)

# Batch response
queries = ["What is Kubernetes?", "What is Docker?", "What is container orchestration?"]

answers = chain.batch(queries)

for answer in answers:
    print(answer)
    print("\n\n\n---------------------------------------------------")
