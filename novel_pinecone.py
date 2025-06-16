import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # Changed from UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
print("Environment variables loaded")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone initialized")

# PDF directory setup
pdf_directory = "./dbms_pdf"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files to process")

# Document processing setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
embeddings = OpenAIEmbeddings()

# Create or connect to index
index_name = os.getenv("PINECONE_INDEX_NAME", "rag-pdfs")

if index_name not in pc.list_indexes().names():
    print(f"Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully")
else:
    print(f"Using existing index '{index_name}'")

# Process each PDF
def process_pdfs():
    docs = []
    
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing {pdf_file}...")
            file_path = os.path.join(pdf_directory, pdf_file)
            loader = PyPDFLoader(file_path)  # Using PyPDFLoader instead
            raw_docs = loader.load()
            print(f"Loaded {len(raw_docs)} pages")
            
            split_docs = text_splitter.split_documents(raw_docs)
            print(f"Split into {len(split_docs)} chunks")
            
            # Add source metadata
            for doc in split_docs:
                doc.metadata["source"] = pdf_file
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1  # Make pages 1-indexed
            
            docs.extend(split_docs)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return docs

# Main execution
if __name__ == "__main__":
    # First uninstall problematic packages
    print("\n⚠️ Removing potentially conflicting packages...")
    os.system("pip uninstall pdfminer pdfminer.six -y")
    print("Installing required packages...")
    os.system("pip install pypdf")
    
    # Process all PDFs
    all_docs = process_pdfs()
    print(f"\nTotal documents processed: {len(all_docs)}")
    
    # Store in Pinecone
    if all_docs:
        print(f"\nUploading to Pinecone index '{index_name}'...")
        PineconeVectorStore.from_documents(
            documents=all_docs,
            embedding=embeddings,
            index_name=index_name
        )
        print("Documents successfully stored in Pinecone!")
        
        # Verify
        index_stats = pc.describe_index(index_name)
        print(f"\nIndex Stats:")
        print(f"- Dimension: {index_stats.dimension}")
        print(f"- Metric: {index_stats.metric}")
        print(f"- Status: {index_stats.status.state}")
    else:
        print("⚠️ No documents were processed - check your PDF files and error messages")