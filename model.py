import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

OPENAI_API_KEY = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

prompt = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template="""You are an AI assistant. You should answer all questions.
    chat_history:{chat_history}
    Human: {question}

    AI:"""
)

llm = ChatOpenAI(model="gpt-4", temperature=0.2)

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)

llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

st.set_page_config(
    page_title="My ChatBOT App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ChatGPT Clone")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, ask me any question."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )
    with st.chat_message("user"):
        st.write(user_prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt, chat_history=memory.buffer)
            st.write(ai_response)

            new_ai_message = {
                "role": "assistant",
                "content": ai_response
            }
            st.session_state.messages.append(new_ai_message)
