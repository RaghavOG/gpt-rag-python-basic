
## 🔁 OVERALL GOAL

* Query PDFs you've embedded (like Indian Evidence Act, Criminal Code, etc.)
* Use OpenAI + Pinecone to **retrieve relevant content**
* Use LLM to answer based on retrieved chunks
* Show sources with page numbers and short excerpts
* Keep **contextual session history** for smart follow-ups

---

## ⚙️ STEP-BY-STEP FLOW

---

### 1. **Environment Setup**

```python
load_dotenv()
```

Loads your `.env` containing:

* `OPENAI_API_KEY`
* `PINECONE_API_KEY`
* `PINECONE_INDEX_NAME`

✅ Preps keys for usage.

---

### 2. **Initialize Pinecone**

```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), timeout=30)
```

Connects to Pinecone using your API key.
`timeout=30` prevents long hangs if the service lags.

---

### 3. **Load Embeddings**

```python
embeddings = OpenAIEmbeddings()
```

Initializes OpenAI embedding model (`text-embedding-3-small`) — same one used during PDF upload.

---

### 4. **Initialize LLM**

```python
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
```

Uses a **cheap, fast LLM** (GPT-4.1-nano). You can later upgrade to `gpt-4`, `gpt-4o`, etc.

---

### 5. **Connect to Existing Pinecone Index**

```python
vectorstore = PineconeVectorStore.from_existing_index(...)
```

Tries 3 times to connect to your index (e.g. `"rag-pdfs"`).
If it fails, it retries with **exponential backoff** (`2^n` seconds wait).

---

### 6. **Create RetrievalQA Chain**

```python
qa = RetrievalQA.from_chain_type(...)
```

This creates the actual **question-answer chain**:

* **Retriever**: MMR (Max Marginal Relevance) for diverse chunk retrieval
* **Search params**: `k=5`, `score_threshold=0.7`
* **Chain type**: `"stuff"` (stuff chunks into prompt and pass to LLM)
* **LLM**: `ChatOpenAI` model
* **Returns**: Final answer + source documents

---

### 7. **Interactive Chat Loop**

```python
while True:
    query = input("❓ Your question: ").strip()
```

Keeps taking user input until `exit` or `quit`.

---

### 8. **Contextual Query Formation**

```python
contextual_query = (
    f"Previous questions: {session_history[-3:]}...\n"
    f"Current question: {query}"
)
```

This helps with **multi-turn reasoning**. The LLM knows your recent context.

---

### 9. **Answer Retrieval + Display**

```python
result = qa.invoke({"query": contextual_query})
```

Runs your query through:

* Vector search
* Chunk stuffing
* LLM generation

Then this is passed to `display_answer()` which:

* ✅ Shows answer in **green**
* 📘 Shows sources in **blue**
* 📄 Shows short excerpts from PDF pages in **cyan**

---

### 10. **Session Memory & Timing**

```python
session_history.append(query)
print(f"\n⏱️ Response time: {elapsed:.2f}s")
```

* Logs your question history
* Tracks time to respond

---

### 11. **Exit & Summary**

```python
print("\n📝 Session summary:")
```

When you exit, it prints all questions you asked.

---

## 🧠 WHAT YOU'VE BUILT

An end-to-end **PDF-based Law Q\&A system**:

* 📥 Chunk & store: ✅ Done earlier
* 🔍 Ask questions: ✅ Done here
* 🔗 Source traceability: ✅ Pages + filenames shown
* 📈 Smart context: ✅ Uses history for better answers
* 🧼 Robust: ✅ Retries, fallbacks, and error handling

