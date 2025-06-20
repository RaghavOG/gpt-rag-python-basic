from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

print("Generating detailed report on 'black hole'...")
prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)
print("Detailed report generated:\n")
print(result.content)
print("\nGenerating 5 line summary of the report...")
prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)
print("Summary generated:\n")
print(result1.content)
