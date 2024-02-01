from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company tha makes {product}?")
prompt.format(product="colorful socks")

template = "You are a helpful assistant that translaces {input_language} to {output_language}."
human_template="{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chat_prompt.format_messages(input_language="English", output_language="Japanese", text="Hello, how are you?")
