from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal",
#     "Paris is the capital of France"
# ]

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


# result_With_LLM = llm.invoke("What is the capital of India")
# result_With_Model = model.invoke("What is the capital of India")
# result_with_embedding = embedding.embed_query("What is the capital of India")


# print("Result with LLM:", result_With_LLM)
# print("Result with Model:", result_With_Model)
# print("*****************************************")
# print("Result with Embedding:", result_with_embedding)
# print("*****************************************")

# result = embedding.embed_documents(documents)
# print(str(result))


query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

