from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

load_dotenv('.env')

output_parser = CommaSeparatedListOutputParser()
chat_model = ChatOpenAI()

template = "Generate a list of 5 {text}.\n\n{format_instructions}"
chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
chain = chat_prompt | chat_model | output_parser

result1 = chain.invoke({"text": "colors"})
print(result1)
result2 = chain.invoke({"text": "foods"})
print(result2)
result3 = chain.invoke({"text": "magnificent humans"})
print(result3)