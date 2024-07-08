import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
import os

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

prompt = PromptTemplate(
    input_variables=['chat_history', 'pdf_content', 'question'],
    template="""You are an AI assistant. You should answer all questions based on the PDF content and the chat history.
    chat_history: {chat_history}
    PDF Content: {pdf_content}
    Human: {question}

    AI:"""
)

llm = ChatOpenAI(model="gpt-4", temperature=0.2)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="question",
    k=5
)

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

if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_content = extract_text_from_pdf(uploaded_file)
    st.session_state.pdf_content = pdf_content
    st.success("PDF content successfully extracted!")

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
            inputs = {
                "question": user_prompt,
                "chat_history": memory.buffer,
                "pdf_content": st.session_state.pdf_content
            }
            ai_response = llm_chain(inputs)
            response_content = ai_response['text']
            st.write(response_content)

            new_ai_message = {
                "role": "assistant",
                "content": response_content
            }
            st.session_state.messages.append(new_ai_message)
