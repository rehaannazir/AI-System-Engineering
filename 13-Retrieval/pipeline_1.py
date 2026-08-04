import sys
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from ingest import receiver, loading_text
from langchain_core.stores import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

"--------------------------------------------------------------"
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "12-Ingestion"))

required_formats = {
    ".pdf": PyPDFLoader,
    ".md": partial(TextLoader, encoding="utf-8"),
    ".html": partial(TextLoader, encoding="utf-8"),
    ".txt": partial(TextLoader, encoding="utf-8"),
}

load_dotenv()
"--------------------------------------------------------------"

"""
DOCS -> Parent_Chunks -> Child_Chunks -> VectorStore(Chroma) -> DocStore (InMemory) -> ParentRetriever(all prev nodes) ->
QUERY -> Top_k_chunks -> LLM
"""


query = "What are transformers?"
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap=120
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 30
)

documents = receiver("docs")
docs = loading_text(documents, required_formats)

doc_store = InMemoryStore()
vector_store = Chroma(
    collection_name="child_vecs",
    embedding_function=embeddings
)

parent_retriever = ParentDocumentRetriever(
    docstore=doc_store,
    vectorstore=vector_store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)
parent_retriever.add_documents(docs)

parent_retriever.search_kwargs={
    "k" : 3
}

print(parent_retriever.invoke(query))
