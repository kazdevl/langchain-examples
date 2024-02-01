from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.schema import HumanMessage

load_dotenv('.env')
llm = OpenAI()
chat_model = ChatOpenAI()

text = "What wold be a good company name for a company tha makes colorful socks?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))
print(chat_model.invoke(messages))