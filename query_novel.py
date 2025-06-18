from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import time
from colorama import Fore, Style, init  # Added import here

# Load environment variables
load_dotenv()
print("✅ Environment variables loaded")

# Initialize components with error handling
try:
    # Initialize Pinecone with timeout
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        timeout=30  # 30 seconds timeout
    )
    print("✅ Pinecone initialized")
    
    
    embeddings = OpenAIEmbeddings()  # Fallback to default
    print("⚠️ Using default embedding model")
    
    # Initialize LLM with model fallback
    index_name = os.getenv("PINECONE_INDEX_NAME")  # Default index name
 
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # Final fallback
    print("⚠️ Using gpt-4.1-nano model as default")
    
    # Connect to Pinecone index with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            print(f"✅ Connected to Pinecone index '{index_name}'")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"⚠️ Retrying connection to Pinecone ({attempt + 1}/{max_retries})...")
            time.sleep(2 ** attempt)  # Exponential backoff

    # Enhanced QA chain configuration
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7,
                "lambda_mult": 0.5  # MMR diversity parameter
            }
        ),
        return_source_documents=True,
        chain_type="stuff",  # Other options: "map_reduce", "refine"
        verbose=True  # Show chain reasoning
    )

except Exception as e:
    print(f"❌ Initialization failed: {str(e)}")
    exit(1)

# Enhanced display function with coloring
def display_answer(result):
    print(f"\n{Fore.GREEN}📚 Answer:{Style.RESET_ALL}")
    print(result["result"])
    
    print(f"\n{Fore.BLUE}🔍 Sources:{Style.RESET_ALL}")
    seen_sources = set()
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata["source"]
        if source not in seen_sources:
            print(f"{Fore.YELLOW}{i}. {source}{Style.RESET_ALL} (Page {doc.metadata.get('page', 'N/A')})")
            seen_sources.add(source)
            print(f"{Fore.CYAN}   Excerpt: {doc.page_content[:200]}...{Style.RESET_ALL}\n")

# Main interaction loop with session tracking
print("\n" + "="*50)
print(f"{'DBMS PDF Knowledge Assistant':^50}")
print("="*50)
print("Type 'exit' or 'quit' to end session\n")

session_history = []
try:
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Session ended")
                break
            if not query:
                continue
                
            print("🔎 Searching knowledge base...")
            start_time = time.time()
            
            # Add context from session history
            contextual_query = (
                f"Previous questions in this session: {session_history[-3:] if session_history else 'None'}\n"
                f"Current question: {query}"
            )
            
            result = qa.invoke({"query": contextual_query})
            display_answer(result)
            
            # Store successful queries
            session_history.append(query)
            
            # Performance metrics
            elapsed = time.time() - start_time
            print(f"\n⏱️  Response time: {elapsed:.2f}s")
            
        except KeyboardInterrupt:
            print("\n🛑 Session interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue

finally:
    # Session summary
    if session_history:
        print("\n📝 Session summary:")
        for i, q in enumerate(session_history, 1):
            print(f"{i}. {q}")
    print("\n✨ Thank you for using the DBMS Knowledge Assistant!")