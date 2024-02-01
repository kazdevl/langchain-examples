from langchain.output_parsers import CommaSeparatedListOutputParser

p = CommaSeparatedListOutputParser()
print(p.parse("hi, bye, thanks"))