import os
from typing import TypedDict, Annotated, Sequence, Union
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.combine_documents.refine import create_refine_documents_chain
from langchain_core.documents import Document

# Initialize environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ======================
# 1. SETUP COMPONENTS
# ======================

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Initialize Pinecone
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# ======================
# 2. HYBRID RETRIEVER
# ======================

class HybridRetriever:
    def __init__(self, vectorstore, text_splitter):
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter
        self.bm25_retriever = None
        self.all_docs = []
    
    def add_documents(self, documents):
        self.all_docs.extend(documents)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.bm25_retriever = BM25Retriever.from_texts(
            texts, metadatas=metadatas
        )
        self.bm25_retriever.k = 5
    
    def get_relevant_documents(self, query, filters=None):
        # Vector similarity search
        vector_results = self.vectorstore.similarity_search(
            query, 
            k=5, 
            filter=filters
        )
        
        # Keyword search (if documents exist)
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
        
        # Combine and deduplicate
        combined = vector_results + bm25_results
        seen = set()
        deduped = []
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                deduped.append(doc)
        return deduped[:10]

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever(vectorstore, text_splitter)

# ======================
# 3. RESPONSE CHAINS
# ======================

def create_retrieval_chain(chain_type="stuff"):
    # Define prompt
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context:
        
        Context:
        {context}
        
        Question: {input}"""
    )
    
    if chain_type == "refine":
        chain = create_refine_documents_chain(llm, prompt)
    else:
        chain = create_stuff_documents_chain(llm, prompt)
    
    return {
        "context": lambda x: hybrid_retriever.get_relevant_documents(
            x["input"], 
            filters=x.get("filters", {})
        ),
        "input": RunnablePassthrough()
    } | chain

# Create chains
summary_chain = create_retrieval_chain("stuff")
factual_chain = create_retrieval_chain("refine")

# ======================
# 4. LANGRAPH STATE
# ======================

class GraphState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    query_type: str
    filters: dict
    response: str

# ======================
# 5. GRAPH NODES
# ======================

def classify_query(state: GraphState):
    # Classify query type
    classifier_prompt = ChatPromptTemplate.from_template(
        """Classify the query into one of: 
        - 'summary': Requests for summaries or overviews
        - 'case_search': Specific case lookups with filters
        - 'factual': Direct factual questions
        
        Examples:
        User: "Summarize Roe v Wade" -> summary
        User: "Find 2015 Supreme Court cases about free speech" -> case_search
        User: "What's the legal definition of negligence?" -> factual
        
        Current conversation:
        {chat_history}
        
        Query: {input}"""
    )
    
    classifier_chain = classifier_prompt | llm | StrOutputParser()
    query_type = classifier_chain.invoke(state)
    
    # Simple cleanup
    if "summary" in query_type.lower():
        return {**state, "query_type": "summary"}
    elif "case_search" in query_type.lower():
        return {**state, "query_type": "case_search"}
    else:
        return {**state, "query_type": "factual"}

def extract_filters(state: GraphState):
    # Extract metadata filters
    if state["query_type"] != "case_search":
        return {**state, "filters": {}}
    
    filter_prompt = ChatPromptTemplate.from_template(
        """Extract search filters from the query in JSON format. 
        Use only these parameters:
        - court (Supreme, Appellate, District)
        - year (integer)
        - jurisdiction (state or federal)
        
        Examples:
        Query: "Supreme Court cases from 2015" 
        → {{"court": "Supreme", "year": 2015}}
        
        Query: "Recent appellate cases in California"
        → {{"court": "Appellate", "jurisdiction": "California"}}
        
        Query: {input}
        
        Output ONLY JSON:"""
    )
    
    filter_chain = filter_prompt | llm | JsonOutputParser()
    filters = filter_chain.invoke({"input": state["input"]})
    return {**state, "filters": filters}

def route_query(state: GraphState):
    # Route to appropriate chain
    if state["query_type"] == "summary":
        return "summarize"
    elif state["query_type"] == "case_search":
        return "retrieve_case"
    else:
        return "answer_factual"

def summarize_node(state: GraphState):
    response = summary_chain.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"]
    })
    return {**state, "response": response}

def retrieve_case_node(state: GraphState):
    response = factual_chain.invoke({
        "input": state["input"],
        "filters": state["filters"],
        "chat_history": state["chat_history"]
    })
    return {**state, "response": response}

def answer_factual_node(state: GraphState):
    response = factual_chain.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"]
    })
    return {**state, "response": response}

# ======================
# 6. GRAPH CONSTRUCTION
# ======================

workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("classify", classify_query)
workflow.add_node("extract_filters", extract_filters)
workflow.add_node("route", route_query)
workflow.add_node("summarize", summarize_node)
workflow.add_node("retrieve_case", retrieve_case_node)
workflow.add_node("answer_factual", answer_factual_node)

# Define edges
workflow.set_entry_point("classify")
workflow.add_edge("classify", "extract_filters")
workflow.add_edge("extract_filters", "route")

workflow.add_conditional_edges(
    "route",
    route_query,
    {
        "summarize": "summarize",
        "retrieve_case": "retrieve_case",
        "answer_factual": "answer_factual"
    }
)

workflow.add_edge("summarize", END)
workflow.add_edge("retrieve_case", END)
workflow.add_edge("answer_factual", END)

# Compile graph
rag_agent = workflow.compile()

# ======================
# 7. UTILITY FUNCTIONS
# ======================

def ingest_documents(file_path: str):
    """Add documents to the retrieval system"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    
    # Add to vectorstore
    Pinecone.from_documents(
        chunks, 
        embeddings, 
        index_name=PINECONE_INDEX_NAME
    )
    
    # Add to BM25
    hybrid_retriever.add_documents(chunks)

def invoke_agent(query: str, session_id: str):
    """Execute the agent with memory management"""
    # Load conversation history
    memory.load_memory_variables({})
    memory.save_context(
        {"input": query}, 
        {"output": ""}
    )
    chat_history = memory.buffer
    
    # Execute agent
    result = rag_agent.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    # Update memory
    memory.save_context(
        {"input": query},
        {"output": result["response"]}
    )
    
    return result["response"]

# ======================
# 8. USAGE EXAMPLE
# ======================

# Ingest documents (do once)
# ingest_documents("legal_document.pdf")

# Run queries
response = invoke_agent(
    "Summarize key points from the Miranda rights cases",
    "user_session_123"
)
print(response)

response = invoke_agent(
    "Find Supreme Court cases from 2015 about free speech",
    "user_session_123"
)
print(response)