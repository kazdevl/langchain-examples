import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def init_page():
    st.set_page_config(page_title="WebSite Summarizer", page_icon="🔗")
    st.header("WebSite Summarizer 🔗")
    st.sidebar.title("Menu")

def init_massages():
    clear_button = st.sidebar.button("Clear conversations", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful assistant that translates English to Japanese.",
            ),
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Select model:", ["GPT-3.5", "GPT-4"])
    if model == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    return ChatOpenAI(model_name=model, temperature=0)

def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

def validate_url(url):
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching content..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write("Failed to fetch content from the URL.")
        return None

def build_propmt(content, n_chars=300):
    return f"""以下はとある、webページのコンテンツです。内容を{n_chars}程度でわかりやすく要約してください。
======
{content[:1000]}

======

日本語で書いてね!
"""

def get_answer(llm,messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def main():
    load_dotenv('.env')
    init_page()
    init_massages()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please input valid url")
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_propmt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None
    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: {sum(costs):.5f}**")
    if len(costs) == 0:
        st.sidebar.markdown(f"**Average cost: 0.0**")
    else:
        st.sidebar.markdown(f"**Average cost: {sum(costs)/len(costs):.5f}**")
    st.sidebar.markdown("### Each Cost")
    for i, cost in enumerate(costs):
        st.sidebar.markdown(f"**{i+1}番目: {cost:.5f}**")

if __name__ == "__main__":
    main()
