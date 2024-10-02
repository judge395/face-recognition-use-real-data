import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")


llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")


# Start the chat
print("Welcome to the chat! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    result = llm.invoke(user_input)
    print(f"Assistant: {result.content}")