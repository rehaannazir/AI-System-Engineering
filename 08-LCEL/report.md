# Naive RAG vs LangChain RAG

| Step | Naive RAG | LangChain |
|------|-----------|-----------|
| **LLM** | `client = genai.Client(api_key=...)`<br>`client.models.generate_content(...)` | `llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")`<br>`llm.invoke(prompt)` |
| **Embedding** | `client.models.embed_content(...)` | `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`<br>`embed_documents()`<br>`embed_query()` |
| **Document Loading** | `fitz.open()` | Same (or use LangChain Loaders later) |
| **Chunking** | Manual loop | Same (or `RecursiveCharacterTextSplitter` later) |
| **Vector Store** | Manually create vectors and search | `FAISS.from_texts(texts=doc_chunks, embedding=embedding)` |
| **Retriever** | Manual similarity search | `retriever = vector_store.as_retriever()` |
| **Context Formatting** | Manual loop | `RunnableLambda(format_docs)` |
| **Prompt** | `f"""...{query}...{context}..."""` | `ChatPromptTemplate.from_template(...)` |
| **Pass Query** | Manual variable passing | `RunnablePassthrough()` |
| **Pipeline** | Call every function manually | Connect Runnables using `|` |
| **Output Parser** | `response.text` / manual | `StrOutputParser()` |
| **Execution** | Every function one by one | `chain.invoke(query)` |
| **Streaming** | Provider-specific | `chain.stream(query)` |
| **Batch** | `for` loop | `chain.batch(queries)` |
| **Async** | Manual async code | `chain.ainvoke(query)` |

---

# Manual Naive RAG

```python
context = retriever(query)

prompt = create_prompt(query, context)

response = llm(prompt)

answer = parser(response)

print(answer)
```

---

# LangChain RAG

```python
chain = (
    RunnableParallel(
        query=RunnablePassthrough(),
        context=retriever | RunnableLambda(format_docs)
    )
    | prompt
    | llm
    | parser
)

answer = chain.invoke(query)
```

---

# Why LangChain?

✅ Standard API for all LLMs

✅ Standard API for all Embedding Models

✅ Standard API for all Vector Stores

✅ Components become Runnables

✅ Connect components with `|`

✅ Easy Streaming (`.stream()`)

✅ Easy Batch (`.batch()`)

✅ Easy Async (`.ainvoke()`)

✅ Modular & Reusable

---

# Mental Model

Naive RAG

```
Function 1
    ↓
Function 2
    ↓
Function 3
    ↓
Function 4
```

LangChain

```
Runnable 1
      |
Runnable 2
      |
Runnable 3
      |
Runnable 4
```

---

# Golden Rule

**Naive RAG = You manually call every function.**

**LangChain = You build a pipeline once, then execute it using `.invoke()`, `.stream()`, `.batch()`, or `.ainvoke()`.**