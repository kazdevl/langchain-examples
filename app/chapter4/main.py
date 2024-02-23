from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.callbacks import get_openai_callback

def init_page():
    st.set_page_config(page_title="Langchain", page_icon="🔗")
    st.header("Langchain")
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

    temperature = st.sidebar.slider("Tempreature:", min_value=0.0, max_value=2.0, value=0.1, step=0.01)

    return ChatOpenAI(model=model, temperature=temperature)

def get_answer(llm, messages):
    with get_openai_callback() as callback:
        answer = llm(messages)
    return answer.content, callback.total_cost

def main():
    load_dotenv('.env')
    init_page()
    init_massages()

    llm = select_model()

    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

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