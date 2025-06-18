
## 🔁 **OVERALL FLOW**

You're taking **multiple PDF law files**, **splitting** their content into chunks, **embedding** them using OpenAI, and storing them in **Pinecone** — a vector DB.
Later, you’ll use those embeddings to **retrieve relevant text** for any question you ask.

---

## 🔧 STEP-BY-STEP EXPLANATION

### 1. **Environment Setup**

```python
load_dotenv()
```

Loads your `.env` file, which contains:

* `PINECONE_API_KEY`
* `PINECONE_INDEX_NAME`
* `OPENAI_API_KEY`

---

### 2. **Pinecone Initialization**

```python
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
```

This connects you to Pinecone using your API key.

---

### 3. **PDF Directory Discovery**

```python
pdf_directory = "./dbms_pdf"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
```

Finds **all PDF files** in a folder. You can keep your law PDFs there — like:

* The Indian Evidence Act.pdf
* The Code of Civil Procedure.pdf
* etc.

---

### 4. **Chunking Strategy Setup**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```

You split each PDF page into chunks of 1000 characters with 200 characters overlapping — this helps preserve context between chunks.

---

### 5. **Embedding Model Setup**

```python
embeddings = OpenAIEmbeddings()
```

This loads OpenAI’s embedding model (`text-embedding-3-small` or similar) to convert text into vector form.

---

### 6. **Pinecone Index Handling**

```python
if index_name not in pc.list_indexes().names():
    pc.create_index(...)
```

Checks if the index exists in Pinecone. If not, it creates one with:

* **Dimension = 1536** (required by OpenAI embeddings)
* **Metric = cosine**
* **Serverless spec** with AWS region `us-east-1`

---

### 7. **PDF Loading and Chunking**

```python
loader = PyPDFLoader(file_path)
raw_docs = loader.load()
split_docs = text_splitter.split_documents(raw_docs)
```

For each PDF:

* Reads all pages
* Splits them into overlapping chunks
* Attaches metadata: `filename` + `page number`

---

### 8. **Add Metadata**

```python
doc.metadata["source"] = pdf_file
doc.metadata["page"] = doc.metadata.get("page", 0) + 1
```

This helps in traceability later — when you answer a query, you can say *"This came from Page 5 of The Indian Evidence Act".*

---

### 9. **Upload to Pinecone**

```python
PineconeVectorStore.from_documents(...)
```

Now that you have a list of all chunks:

* Embeds them
* Pushes them into the `rag-pdfs` Pinecone index

---

### 10. **Verify**

```python
index_stats = pc.describe_index(index_name)
```

Gets index info like:

* Status: Ready / Initializing
* Metric
* Vector dimension

---

## ✅ FINAL THOUGHTS

Once this script runs:

* Your law PDFs are **fully chunked, embedded, and stored**
* You can now build a **retrieval-augmented app** where:

  * You ask *"What is Section 27 of the Evidence Act?"*
  * It retrieves the most similar vector chunk
  * That chunk is passed to an LLM for answering.


