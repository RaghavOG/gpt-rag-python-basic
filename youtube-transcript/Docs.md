Based on the provided code, here's a structured breakdown of the main functions/methods used, their parameters, default values, and usage context:

---

### **1. `YouTubeTranscriptApi.get_transcript()`**
**Purpose**: Fetch YouTube video captions.  
**Parameters**:  
- `video_id` (str): *Required*. YouTube video ID (e.g., `"Gfr50f6ZBvo"`).  
- `languages` (list[str]): *Optional*. Language codes for captions (default: `None` → all languages).  
  - Passed in code: `languages=["en"]` (English only).  
- `proxies` (dict): *Optional*. Proxy configuration (default: `None`).  
- `cookies` (dict): *Optional*. Browser cookies for restricted videos (default: `None`).  

**Code Usage**:  
```python
YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
```

---

### **2. `RecursiveCharacterTextSplitter.create_documents()`**
**Purpose**: Split text into chunks.  
**Parameters**:  
- `texts` (list[str]): *Required*. List of text strings to split.  
  - Passed in code: `[transcript]` (single transcript string).  
- `chunk_size` (int): *Optional*. Max characters per chunk (default: `4000`).  
  - Passed in code: `chunk_size=1000`.  
- `chunk_overlap` (int): *Optional*. Overlap between chunks (default: `200`).  
  - Passed in code: `chunk_overlap=200`.  
- `length_function` (Callable): *Optional*. Function to calculate text length (default: `len`).  

**Code Usage**:  
```python
splitter.create_documents([transcript])
```

---

### **3. `OpenAIEmbeddings()` (Constructor)**
**Purpose**: Generate text embeddings.  
**Parameters**:  
- `model` (str): *Optional*. Embedding model ID (default: `"text-embedding-ada-002"`).  
  - Passed in code: `model="text-embedding-3-small"`.  
- `api_key` (str): *Optional*. OpenAI API key (default: `None` → uses environment variable `OPENAI_API_KEY`).  
- `max_retries` (int): *Optional*. Retry attempts on failure (default: `2`).  

**Code Usage**:  
```python
OpenAIEmbeddings(model="text-embedding-3-small")
```

---

### **4. `FAISS.from_documents()`**
**Purpose**: Create a vector store from documents.  
**Parameters**:  
- `documents` (list[Document]): *Required*. Text chunks to index.  
  - Passed in code: `chunks` (from text splitter).  
- `embedding` (Embeddings): *Required*. Embedding model instance.  
  - Passed in code: `embeddings` (OpenAIEmbeddings object).  
- `ids` (list[str]): *Optional*. Custom IDs for documents (default: `None` → auto-generated).  

**Code Usage**:  
```python
FAISS.from_documents(chunks, embeddings)
```

---

### **5. `FAISS.as_retriever()`**
**Purpose**: Create a document retriever from the vector store.  
**Parameters**:  
- `search_type` (str): *Optional*. Search method (`"similarity"`, `"mmr"`, etc.; default: `"similarity"`).  
  - Passed in code: `search_type="similarity"`.  
- `search_kwargs` (dict): *Optional*. Additional search parameters.  
  - Passed in code: `{"k": 4}` → return top 4 documents.  
- `filters` (dict): *Optional*. Metadata filters (default: `None`).  

**Code Usage**:  
```python
vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

---

### **6. `ChatOpenAI()` (Constructor)**
**Purpose**: Initialize OpenAI chat model.  
**Parameters**:  
- `model` (str): *Optional*. Model ID (default: `"gpt-3.5-turbo"`).  
  - Passed in code: `model="gpt-4.1-nano"`.  
- `temperature` (float): *Optional*. Creativity control (default: `0.7`).  
  - Passed in code: `temperature=0.2`.  
- `api_key` (str): *Optional*. OpenAI API key (default: `None` → uses environment variable).  
- `max_tokens` (int): *Optional*. Max tokens to generate (default: `None`).  

**Code Usage**:  
```python
ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
```

---

### **7. `PromptTemplate()` (Constructor)**
**Purpose**: Create a reusable prompt template.  
**Parameters**:  
- `template` (str): *Required*. Prompt structure with variables.  
- `input_variables` (list[str]): *Required*. Variables to inject into the template.  
  - Passed in code: `input_variables=['context', 'question']`.  
- `template_format` (str): *Optional*. Format (e.g., `"f-string"`; default: `"f-string"`).  

**Code Usage**:  
```python
PromptTemplate(
    template="...", 
    input_variables=['context', 'question']
)
```

---

### **8. Chain Components (`RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`)**
**Purpose**: Build processing pipelines.  
**Key Parameters**:  
- **`RunnableParallel`**: Dict of runnables (e.g., `{'context': ..., 'question': ...}`).  
- **`RunnablePassthrough`**: Passes input unchanged (no parameters).  
- **`RunnableLambda`**: Wraps a function (e.g., `RunnableLambda(format_docs)`).  

**Code Usage**:  
```python
RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
```

---

### **9. `StrOutputParser()`**
**Purpose**: Parse LLM output into a string.  
**Parameters**: None.  

**Code Usage**:  
```python
parser = StrOutputParser()
```

---

### **10. Chain Execution (`.invoke()`)**
**Purpose**: Execute a LangChain runnable.  
**Parameters**:  
- `input` (any): *Required*. Input to the chain (e.g., query string).  

**Code Usage**:  
```python
retriever.invoke(test_query)          # Single component
parallel_chain.invoke('who is Demis') # Sub-chain
main_chain.invoke('Summarize video')  # Full pipeline
```

---

### **11. `llm.invoke()` (LLM Call)**
**Purpose**: Generate text from an LLM.  
**Parameters**:  
- `input` (str|PromptValue): *Required*. Prompt to send to the model.  
- `config` (dict): *Optional*. Runtime settings (e.g., `stop` sequences).  

**Code Usage**:  
```python
llm.invoke(final_prompt)  # final_prompt is a PromptValue object
```

---

### **12. Custom Function: `format_docs()`**
**Purpose**: Format retrieved documents into a single string.  
**Parameters**:  
- `retrieved_docs` (list[Document]): *Required*. List of document objects.  

**Code Usage**:  
```python
format_docs(retrieved_docs)  # Returns concatenated page_content
```

---

### Key Observations:
1. **No Explicit `main()` Function**: The script runs sequentially without a defined `main` function.  
2. **Environment Variables**: Assumes `OPENAI_API_KEY` is set in `.env` (via `dotenv`).  
3. **Error Handling**: Only catches `TranscriptsDisabled` (other errors unhandled).  
4. **LangChain Pipeline**: Uses a composed chain (`main_chain = ... | ... | ...`) for QA.  

Let me know if you need clarification on any specific component!