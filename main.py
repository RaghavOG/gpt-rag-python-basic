import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

print("Starting script execution...")
load_dotenv()
print("Environment variables loaded from .env file")

print("Initializing LLM and embeddings...")
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # Changed to more accessible model
embedding = OpenAIEmbeddings()
print("LLM and embeddings initialized successfully")

# Load & index your document
print("\nLoading documents...")
try:
    loader = TextLoader("./docs/sample.txt")
    print("TextLoader created for ./docs/sample.txt")
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} document(s)")
except Exception as e:
    print(f"Error loading documents: {str(e)}")
    raise

# Create vector store
print("\nCreating vector store...")
try:
    db = Chroma.from_documents(docs, embedding)
    print("Vector store created successfully")
    retriever = db.as_retriever()
    print("Retriever initialized from vector store")
except Exception as e:
    print(f"Error creating vector store: {str(e)}")
    raise

# Chain: RAG pipeline
print("\nSetting up RAG pipeline...")
try:
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True  # Changed to False to avoid multiple outputs
    )
    print("RetrievalQA chain created successfully")
except Exception as e:
    print(f"Error setting up RAG pipeline: {str(e)}")
    raise

@traceable(name="Ask Question")
def ask(query: str):
    print(f"\nProcessing question: '{query}'")
    try:
        result = qa.invoke({"query": query})  # Using invoke() instead of run()
        print("Question processed successfully")
        return result["result"]
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        raise

# Test it
if __name__ == "__main__":
    print("\nStarting interactive session...")
    print("Ask me anything from the document (type 'exit' or 'quit' to end):")
    while True:
        try:
            question = input(">> ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            print("Processing your question...")
            response = ask(question)
            print("\nAnswer:")
            print(response)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt, exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue

print("Script execution completed")