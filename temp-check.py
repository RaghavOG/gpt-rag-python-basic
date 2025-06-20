from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set.") 

#  i want to test the output of chatmodel with different temperature values

# Initialize the LLM with different temperature values
llm_low_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)
llm_medium_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
llm_high_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.9)

# Define a sample prompt

prompt = input("Enter a prompt for the LLM: ")

# Generate responses with different temperature settings
response_low_temp = llm_low_temp.invoke( prompt)
response_medium_temp = llm_medium_temp.invoke( prompt)
response_high_temp = llm_high_temp.invoke( prompt)

# Print the responses
print("Response with low temperature (0.1):", response_low_temp.content)
print("Response with medium temperature (0.5):", response_medium_temp.content)
print("Response with high temperature (0.9):", response_high_temp.content)

