from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set.")




# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# Define multiple prompt templates for different tasks
poem_template = ChatPromptTemplate.from_template(
    "Write a 4-line poem about {topic} in the style of {style}"
)

joke_template = ChatPromptTemplate.from_template(
    "Create a funny joke about {topic} that includes the word '{keyword}'"
)

fact_template = ChatPromptTemplate.from_template(
    "List 3 interesting facts about {topic}"
)

quote_template = ChatPromptTemplate.from_template(
    "Generate an inspirational quote about {topic}"
)

# Create individual chains
poem_chain = LLMChain(llm=llm, prompt=poem_template, output_key="poem")
joke_chain = LLMChain(llm=llm, prompt=joke_template, output_key="joke")
fact_chain = LLMChain(llm=llm, prompt=fact_template, output_key="facts")
quote_chain = LLMChain(llm=llm, prompt=quote_template, output_key="quote")

# Create parallel chain
parallel_chain = RunnableParallel(
    poem=poem_chain,
    joke=joke_chain,
    facts=fact_chain,
    quote=quote_chain
)

# Combine the results
combine_template = ChatPromptTemplate.from_template("""
You are a creative content compiler. Combine these outputs about {topic}:

POEM:
{poem}

JOKE:
{joke}

FACTS:
{facts}

QUOTE:
{quote}

Create a cohesive blog post introduction (3-4 paragraphs) that incorporates all these elements.
""")

final_chain = {
    "topic": lambda x: x["topic"],
    "style": lambda x: x["style"],
    "keyword": lambda x: x["keyword"],
    "poem": lambda x: x["poem"]["poem"],
    "joke": lambda x: x["joke"]["joke"],
    "facts": lambda x: x["facts"]["facts"],
    "quote": lambda x: x["quote"]["quote"]
} | combine_template | llm

# Create the full pipeline
full_pipeline = RunnableParallel(
    parallel_output=parallel_chain,
    topic=lambda x: x["topic"],
    style=lambda x: x["style"],
    keyword=lambda x: x["keyword"]
) | final_chain

# Execute the parallel chain
topic = "space exploration"
style = "Shakespeare"
keyword = "rocket"

result = full_pipeline.invoke({
    "topic": topic,
    "style": style,
    "keyword": keyword
})

print("="*50)
print(f"FINAL OUTPUT FOR TOPIC: {topic.upper()}")
print("="*50)
print(result.content)