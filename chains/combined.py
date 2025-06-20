from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch , RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal, List
import json

# Load environment variables
load_dotenv()

# Initialize models
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# ======================
# 1. PYDANTIC MODELS
# ======================
class SentimentAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='Overall sentiment of feedback')
    emotion: Literal['happy', 'angry', 'frustrated', 'excited', 'disappointed'] = Field(description='Primary emotion detected')
    confidence: float = Field(description='Confidence score 0-1')

class FeatureMention(BaseModel):
    feature: str = Field(description='Product feature mentioned')
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='Sentiment about this feature')
    quote: str = Field(description='Exact quote mentioning the feature')

class FeedbackAnalysis(BaseModel):
    summary: str = Field(description='Brief summary of feedback')
    key_insights: List[str] = Field(description='3-5 key insights')
    suggested_actions: List[str] = Field(description='3-5 suggested actions')

# ======================
# 2. PROMPT TEMPLATES
# ======================
# Simple chain: Basic text transformation
clean_prompt = PromptTemplate(
    template="Clean and normalize this customer feedback:\n{feedback}",
    input_variables=['feedback']
)

# Sequential chain: Sentiment classification
sentiment_parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
sentiment_prompt = PromptTemplate(
    template="Analyze sentiment:\n{clean_text}\n{format_instructions}",
    input_variables=['clean_text'],
    partial_variables={'format_instructions': sentiment_parser.get_format_instructions()}
)

# Parallel chain components
feature_parser = PydanticOutputParser(pydantic_object=List[FeatureMention])
feature_prompt = PromptTemplate(
    template="Extract mentioned features:\n{clean_text}\n{format_instructions}",
    input_variables=['clean_text'],
    partial_variables={'format_instructions': feature_parser.get_format_instructions()}
)

summary_prompt = PromptTemplate(
    template="Summarize this feedback in 3 sentences:\n{clean_text}",
    input_variables=['clean_text']
)

# Conditional chain components
positive_response_prompt = PromptTemplate(
    template="Create a personalized thank you response for this positive feedback:\n{summary}\nCustomer emotion: {emotion}",
    input_variables=['summary', 'emotion']
)

negative_response_prompt = PromptTemplate(
    template="Create an apology and solution for this negative feedback:\n{summary}\nKey issues: {issues}",
    input_variables=['summary', 'issues']
)

neutral_response_prompt = PromptTemplate(
    template="Create a neutral response requesting more details:\n{summary}",
    input_variables=['summary']
)

# Final report chain
report_prompt = PromptTemplate(
    template="""Generate comprehensive feedback analysis report:
    
    Customer Feedback:
    {feedback}
    
    Sentiment Analysis:
    {sentiment}
    
    Feature Analysis:
    {features}
    
    Response Generated:
    {response}
    
    Additional Instructions:
    - Create executive summary
    - Identify 3 product improvement opportunities
    - Suggest 2 customer retention strategies
    """,
    input_variables=['feedback', 'sentiment', 'features', 'response']
)

# ======================
# 3. CHAIN CONSTRUCTION
# ======================
# Simple chain: Text cleaning
clean_chain = clean_prompt | model | parser

# Sequential chain: Sentiment analysis
sentiment_chain = {
    "clean_text": clean_chain
} | sentiment_prompt | model | sentiment_parser

# Parallel chain: Feature extraction and summarization
parallel_chain = RunnableParallel(
    features={
        "clean_text": clean_chain
    } | feature_prompt | model | feature_parser,
    summary={
        "clean_text": clean_chain
    } | summary_prompt | model | parser
)

# Conditional chain: Response generation
response_branch = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', 
        positive_response_prompt.partial(emotion=lambda x: x['sentiment'].emotion) | model | parser),
    
    (lambda x: x['sentiment'].sentiment == 'negative', 
        negative_response_prompt.partial(issues=lambda x: "\n- ".join(
            [f"{f.feature}: {f.sentiment}" for f in x['features']]
        )) | model | parser),
    
    neutral_response_prompt | model | parser
)

# Full analysis chain
analysis_chain = {
    "feedback": lambda x: x['feedback'],
    "sentiment": sentiment_chain,
    "features": lambda x: x['parallel']['features'],
    "summary": lambda x: x['parallel']['summary'],
    "response": lambda x: response_branch.invoke({
        "sentiment": x['sentiment'],
        "features": x['features'],
        "summary": x['summary']
    })
} | report_prompt | model | parser

# ======================
# 4. MASTER CHAIN
# ======================
master_chain = {
    "feedback": RunnablePassthrough(),
    "clean_text": clean_chain,
    "sentiment": sentiment_chain,
    "parallel": parallel_chain,
} | {
    "feedback": lambda x: x['feedback'],
    "sentiment": lambda x: x['sentiment'],
    "features": lambda x: x['parallel']['features'],
    "summary": lambda x: x['parallel']['summary'],
    "response": response_branch
} | analysis_chain

# ======================
# 5. EXECUTION
# ======================
feedback_samples = [
    "I absolutely LOVE the new camera features! The night mode is incredible, though battery life could be better. Overall amazing phone!",
    "The latest update completely broke my Bluetooth connectivity. Can't connect to my car or headphones. Very frustrating experience!",
    "The phone looks nice but I'm not sure about the new interface. It's different from what I'm used to."
]

for i, feedback in enumerate(feedback_samples):
    print(f"\n{'='*50}\nANALYZING FEEDBACK #{i+1}\n{'='*50}")
    print(f"Input: {feedback}\n")
    
    result = master_chain.invoke({"feedback": feedback})
    
    print(f"\n{'='*50}\nFINAL REPORT #{i+1}\n{'='*50}")
    print(result)
    print("\n\n")