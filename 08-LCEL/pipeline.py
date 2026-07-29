import fitz
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

# Loading values from env
load_dotenv()

# Making LLM model runnable
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Making Embedding Model
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Loading text from DOC
doc = fitz.open("06-Naive RAG/container_orchestr.pdf")
doc_text = "".join(page.get_text() for page in doc)

# Generating chunks from text
start = 0
overlap = 100
chunk_size = 500
doc_chunks = []

while start < len(doc_text):
    end = start + chunk_size
    doc_chunks.append(doc_text[start:end])
    start = end - overlap

# Getting user  query
query = input("\n Enter your query: ")
query_vector = embedding.embed_query(query)

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


answer = chain.invoke(query)
print(answer)
