import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# In memory (Chat save)
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")


llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# maintain a session id
session_id = str(uuid.uuid4())

# Set up message history store
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(llm, get_session_history)

def generate_response(prompt, last_message=False):
    
    conversation_history = get_session_history(session_id)

    if last_message:
        return conversation_history

    config = {"configurable": {"session_id": session_id}}

    response = with_message_history.invoke(
        [HumanMessage(content=prompt)],
        config=config,
    )
    print("------------------------------------------------------------------")
    print("INPUT TOKEN :: ", response.usage_metadata.get("input_tokens"))
    print("RESPONSE TOKEN :: ", response.usage_metadata.get("output_tokens"))
    print("------------------------------------------------------------------")
    print("\n\n")
    
    return response


# Start the chat
print("Welcome to the chat! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        history =  generate_response(prompt=user_input, last_message=True)
        break
    
    result = generate_response(prompt=user_input)
    print(f"Assistant: {result.content}")

print("<-------------- History -------------->")
print(history)