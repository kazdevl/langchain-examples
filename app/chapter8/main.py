import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from urllib.parse import urlparse

def init_page():
    st.set_page_config(page_title="Youtube Summarizer", page_icon="🔗")
    st.header("Youtube Summarizer 🔗")
    st.sidebar.title("Menu")
    st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    elif model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    else:
        st.session_state.model_name = "gpt-4"

    # 300: 本文以外の指示のtoken数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

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
    with st.spinner("Fetching content..."):
        loader = YoutubeLoader.from_youtube_url(url,add_video_info=True,language=['en','ja'])
        text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size=st.session_state.max_token,
            chunk_overlap=0.1,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        return loader.load_and_split(text_splitter=text_spliter)

def summarize(llm, docs):
    prompt_template="""Write a concise Japanese summary of the following transcipt of Youtbe Video Correctly.
    =====================

    {text}

    =====================

    ここから日本語で書いてね！
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
        response = chain.run(
            {
                "input_documents": docs,
                "token_max": st.session_state.max_token
            }
        )
        print("***結果: ", response)

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
