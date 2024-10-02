# Import necessary libraries
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up the API key for ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Create an agent that uses both the ChatGroq model and the DuckDuckGo search tool
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent with a query
response = agent.invoke({
    "input": "Who is the current captain of Bangladesh Cricket Team ?"
})

# Print the response
print(response)