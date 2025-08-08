import getpass
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from tabulate import tabulate


# set API key for gemini
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# load the gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)


# load csv dataset for RAG generation using Pandas (zero-shot prompting)
df = pd.read_csv("question_answer.csv")
agent =  create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="zero-shot-react-description",
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True
)

# generate responses
agent.invoke("what is data science?")
agent.invoke("What are the key steps in the data science process?")
agent.invoke("What is the difference between supervised and unsupervised learning?")
agent.invoke("Explain the bias-variance tradeoff.")
agent.invoke("what is batman?")
agent.invoke("what is photosynthesis?")














