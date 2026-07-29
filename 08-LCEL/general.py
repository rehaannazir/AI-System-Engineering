from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loading the env variables

load_dotenv()

# Converting normal vars into runnables

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_template("""
You are a teacher and mentor of students.

Answer the following question in a formal manner:
question  {query}

Answer precisely with accurate example
""")

parser = StrOutputParser()

# Calling the runnables and checking mannually

prompt = prompt_template.invoke({"query": "What's the LLM Integration?"})
print(f"prompt : {prompt}")

response = llm.invoke(prompt)
print(f"response : {response}")

structured_output = parser.invoke(response)
print(f"structured_output : {structured_output}")
