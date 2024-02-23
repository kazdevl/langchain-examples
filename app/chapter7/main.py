import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
from urllib.parse import urlparse

def init_page():
    st.set_page_config(page_title="Youtube Summarizer", page_icon="🔗")
    st.header("Youtube Summarizer 🔗")
    st.sidebar.title("Menu")
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

def get_document(url):
    try:
        loader = YoutubeLoader.from_youtube_url(url,add_video_info=True,language=['en','ja'])
    except:
        st.write("Failed to fetch content from the URL.")
        return None
    return loader.load()

def summarize(llm, docs):
    print(docs)
    prompt_template="""Write a concise Japanese summary of the following transcipt of Youtbe Video.
    =====================

    {text}

    =====================

    ここから日本語で書いてね！
    必ず3段落以内の200文字以内で簡潔にまとめること!
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(llm, chain_type="stuff",prompt=prompt)
        response = chain.run(docs)

    return response, cb.total_cost


def main():
    load_dotenv('.env')
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please input valid url")
            output_text = None
        else:
            document = get_document(url)
            with st.spinner("Summarizing..."):
                output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

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
