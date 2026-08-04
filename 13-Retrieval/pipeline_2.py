import sys
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

"--------------------------------------------------------------"
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "12-Ingestion"))
from helper import MultiQuery  # noqa: E402
from ingest import receiver, loading_text  # noqa: E402

required_formats = {
    ".pdf": PyPDFLoader,
    ".md": partial(TextLoader, encoding="utf-8"),
    ".html": partial(TextLoader, encoding="utf-8"),
    ".txt": partial(TextLoader, encoding="utf-8"),
}

load_dotenv()
"--------------------------------------------------------------"

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
docs = loading_text(receiver("docs"), required_formats)

vecstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embedding
)
docstore = InMemoryStore()

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 120
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 30
)

parent_retriever = ParentDocumentRetriever(
    docstore=docstore,
    vectorstore=vecstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

parent_retriever.add_documents(docs)
parent_retriever.search_kwargs = {
    "k" : 3,
    "fetch_k" : 20,
    "lambda_mult" : 0.5
}
parent_retriever.search_type = "mmr"


query = "How can we make multiagents?"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash" )
structured_llm = llm.with_structured_output(MultiQuery)

multi_query_response= structured_llm.invoke(
    F"Generate 5 alternative search queries for '{query}' including original one also. Make whose context and required answer should be same but wording may be different "
)

multi_query = multi_query_response.queries

retrieved_chunks = []

for i in multi_query:

    response = parent_retriever.invoke(i)

    for doc in response:

        retrieved_chunks.append(doc.page_content)

retrieved_chunks = set(retrieved_chunks)
retrieved_chunks = list(retrieved_chunks)
