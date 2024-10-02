from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt_template=PromptTemplate.from_template(
    """
    Translate the {input} into {language} language
    """
).format(input="How are you ?" , language="bengali")

response = llm.invoke(prompt_template)

print(response.content)



