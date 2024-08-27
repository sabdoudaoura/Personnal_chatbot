#####################################################################################################################################################################
############################### This script performs RAG and display result via a streamlit webapp ##################################################################
#####################################################################################################################################################################
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import chromadb
import os
import streamlit as st
from settings import CHROMA_SETTINGS
from gpt_quest import QA_api, QA_api_w_memory, QA_local, response_generator, embeddings, llm
from langchain_core.messages import AIMessage, HumanMessage

if not load_dotenv():
    print(".env file misssing or empty.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4)) 
memory = False

# Set sources visibility
if "show_sources" not in st.session_state:
    st.session_state["show_sources"] = False

# Set memory
if "memory" not in st.session_state:
    st.session_state["memory"] = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_conversation():
  st.session_state.messages = []
def print_sources():
  st.session_state.show_sources = True
  st.session_state.messages = []
def remove_sources():
  st.session_state.show_sources = False
  st.session_state.messages = []
  

if model_type != "Mistral local":
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    if memory: 
        chain = QA_api_w_memory(chroma_client, db, llm=llm)
        st.write("Memory is activated")
    else:
        chain = QA_api(chroma_client, db, llm=llm)


st.title("🤖 Personal chatbot")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# "with" notation
with st.sidebar:
    st.title("Sidebar")
    st.button("Print sources", on_click=print_sources)
    st.write('')
    st.write('')
    st.button("Hide sources", on_click=remove_sources)  
    memory = st.toggle("Activate memory", False)
    if memory:
        st.write("Memory activated!")
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')       
    st.button("Clear chat history", on_click=reset_conversation)
chat_history = []

# Accept user input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if model_type == "Mistral local":

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

           
            answer, docs =  QA_local(prompt, embeddings=embeddings)

            sources = " "
            for i, document in enumerate(docs):
                sources = sources + f"\n source {i+1}: " + document.metadata["source"] +" page "+ str(document.metadata["page"]) + document.page_content

            if st.session_state["show_sources"]:
                stream = response_generator(answer +'-------------------------'+ str(sources))
            else : 
                stream = response_generator(answer) 

            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})


    else : 
    # Display assistant response in chat message container
        with st.chat_message("assistant"):
            
            if memory :
                st.write("Memory is activated")
                res = chain.invoke({"input": prompt, "memory": st.session_state.messages})
                chat_history.extend(
                [
                    HumanMessage(content=prompt),
                    AIMessage(content=res['answer']),
                ]
            )
            else :
                res = chain.invoke({"input": prompt})
            answer, docs = res['answer'], [] if not st.session_state["show_sources"] else res['context']

            sources = ""
            for i, document in enumerate(docs):
                sources = sources + f"\n source {i+1}: " + document.metadata["source"] +" page "+ str(document.metadata["page"] + 1) + document.page_content

            if st.session_state["show_sources"]:
                stream = response_generator(answer +'-----------------------------'+ sources)
            else : 
                stream = response_generator(answer) 

            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})