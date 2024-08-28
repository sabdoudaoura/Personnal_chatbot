#####################################################################################################################################################################
############################### This script performs RAG and display result via a streamlit webapp ##################################################################
#####################################################################################################################################################################
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.memory import ConversationBufferMemory
#from langchain.memory import ConversationSummaryMemory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pathlib import Path
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import streamlit as st
import chromadb
import os
import time

#from chromadb import Chroma, PersistentClient
if not load_dotenv():
    print(".env file misssing or empty.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4)) 
temperature = float(os.environ.get('TEMPERATURE', 0.5))
mistral_models_path = Path.cwd().joinpath('mistral_models')
token_hf = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
model_name = str(os.environ.get("MISTAL_LOCAL_NAME", None))

from settings import CHROMA_SETTINGS

# Define the embedding model
match embeddings_model_name:
    case "OpenAI api":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    case "Mistral api":
        # Define the embedding model
        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.environ.get("MISTRAL_API_KEY"))
    
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Embedding model {embeddings_model_name} is not supported. Please choose one of the following: Open AI api, Mistral local, Mistral api")

# Prepare the LLM
match model_type:
    case "Mistral api":
        llm = ChatMistralAI(model="open-mistral-7b", temperature=temperature)
    case "Mistral local":
        mistral_models_path = Path.cwd().joinpath('mistral_models')
        mistral_models_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_name, allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3", "tokenizer_config.json", "tokenizer.model", "tokenizer.json", "special_tokens_map.json", "model.safetensors.index.json", "generation_config.json", "config.json"], local_dir=mistral_models_path, token=token_hf)
    case "GPT3.5 api":
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    case "GPT4o-mini-api":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: Mistral api, Mistral local, GPT3.5 api, GPT4o mini api")


def QA_api(chroma_client, db, llm, embeddings=embeddings):
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = ChatOpenAI()

    system_prompt = (
        "Utilise le contexte donné pour répondre à la question. "
        "Si le contexte ne permet pas de repondre, dis que tu ne sais pas. "
        "Contexte: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain

def QA_api_w_memory(chroma_client, db, llm, embeddings=embeddings):
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = ChatOpenAI()

    system_prompt = (
        "Utilise le contexte donné pour répondre à la question. "
        "Si le contexte ne permet pas de repondre, dis que tu ne sais pas. "
        "Contexte: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

    #memory
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return chain



def QA_local(query, embeddings=embeddings):
    """This function performs a question-answering task using a local Mistral model.

    Args:
        query (str): The answer to answer
        embeddings (_type_, optional): embedding function used for the database. Defaults to embeddings.

    Returns:
        str: answer to the question and the associated context
    """

    # Load the existing Chroma vector store
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # Perform the similarity search
    sources = vectorstore.similarity_search(query, k=target_source_chunks)  

    # Iterate over the results and concatenate the page content
    retrieved_texts = " "
    for result in sources:
        retrieved_texts += result.page_content + "\n"

    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path }/tokenizer.model.v3")
    model = Transformer.from_folder(mistral_models_path)
    input_text = f"Utilise le contexte donné pour répondre à la question. Si le contexte ne permet pas de repondre, dis que tu ne sais pas. Contexte: {retrieved_texts} , Question : {query}"
    
    completion_request = ChatCompletionRequest(messages=[UserMessage(content= input_text)])

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model, max_tokens=1000, temperature=temperature, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])


    return result, sources

# Streamed response emulator
def response_generator(my_string):

    for word in my_string.split():
        yield word + " "
        time.sleep(0.01)

def reset_conversation():
  st.session_state.messages = []
def print_sources():
  st.session_state.show_sources = True
  st.session_state.messages = []
def remove_sources():
  st.session_state.show_sources = False
  st.session_state.messages = []