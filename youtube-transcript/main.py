import dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

video_id = "Gfr50f6ZBvo"  
try:
    print(f"Fetching transcript for video ID: {video_id}")
    transcript_list = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("Transcript fetched successfully:")
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

print("Splitting transcript into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(f"Number of chunks created: {len(chunks)}")

print("Generating embeddings and creating vector store...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created.")

retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4})

print("Testing retriever with a sample query...")
test_query = 'What is deepmind'
retriever.invoke(test_query)
print(f"Retriever invoked with query: '{test_query}'")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
print(f"Retrieving docs for question: '{question}'")
retrieved_docs = retriever.invoke(question)
print(f"Number of docs retrieved: {len(retrieved_docs)}")

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print("Context text prepared for prompt.")

final_prompt = prompt.invoke({"context": context_text, "question": question})
print("Prompt prepared for LLM.")

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

print("Invoking LLM for answer...")
answer = llm.invoke(final_prompt)
print("Answer from LLM:")
print(answer.content)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

print("Testing parallel_chain with query: 'who is Demis'")
parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print("Invoking main_chain with query: 'Can you summarize the video'")
summary = main_chain.invoke('Can you summarize the video')
print("Summary:")
print(summary)
