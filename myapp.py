import streamlit as st
import PyPDF2
import re
import os
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfReader
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import openai

uploaded_file = st.file_uploader("Choose your file", "pdf")
if uploaded_file is not None:
    read_pdf = PdfReader(uploaded_file)
    count = len(read_pdf.pages)
    text = " "
    for i in range(count):
        page = read_pdf.pages[i]
        text += page.extract_text()
        #st.write(text)

# Initialize session state

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


# Function to get user_input

def get_text():
    input_text = st.text_input("You:", st.session_state["input"], key="input",
                               placeholder="Your AI assistant here..! Ask me anything...!",
                               label_visibility="hidden")
    return input_text


# Creating a function to erase previous session history and start new chat

def new_chat():
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    #st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()


st.title("Memory-Bot")

# Input api-key

api = st.sidebar.text_input("Enter your API Key here", type="password")
# os.environ["OPENAI_API_KEY"] = api
MODEL = st.sidebar.selectbox(label="Model",
                             options=['gpt-3.5-turbo', 'text-da-vinci-003', 'text-da-vinci-002', 'code-da-vinci-002'])

if api:
    llm = OpenAI(
        temperature=0,
        openai_api_key=api,
        model_name=MODEL,
    )
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)

    # Create conversation chain

    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )
else:
    st.error("No API found")

st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get the user input

user_input = get_text()

# Generate output using Conv chain object

if user_input:
    output = Conversation.run(input=text + "Use this text to answer the following question" + user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])

# Create a conv memory
