from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env')
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは詩的なアシスタントで、複雑なプログラミングの概念を創造的なセンスで説明することに長けています。"},
        {"role": "user", "content": "プログラミングにおける再帰の概念を説明する詩を作ろう。"},
    ]
)

print(completion.choices[0].message)