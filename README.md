## 🤖 Personal chatbot

This chatbot can answer questions on your on documents. This chatbot has memory and performs hybrid search.
1 Create your own vector database, using pdf, text or word documents
2 Perform similarity search to retrieve the best chunks to answer a question you may have
3 Answer your question in a minimalist interface

## Specification 

Database : The database used is a Chroma database
Embedding model : The embedding model can be either OPENAI or MISTRAL AI api call 
Q&A model : The LLM used to answer the questions can be either an OPENAI model or a MISTRAl model via API. It can also be a MISTRAl 7b model stored locally.
Interface : The interface is built using streamlit

## Installation

1. Install the dependencies: `pip install -r requirements.txt`.
2. Specify the following information in the .env file
The embedding model : EMBEDDINGS_MODEL_NAME among ‘OpenAI api’ and ‘Mistral api’.
The chatbot model: MODEL_TYPE from among ‘Mistral api’, ‘GPT3.5 api’, ‘GPT4o-mini-api’ and ‘Mistral local’.
3. Specify your api key OPENAI_API_KEY or MISTRAL_API_KEY or HuggingFace

! If you decide to change your embedding model, the database will have to be deleted and recreated. 

## Description
Folders 
db : contains the vector database
source_docs : contains the source documents to be incorporated into the vector database. 
mistral_models : contains the Mistral model stored locally and the associated tokenizer. 

Files
vectorizer.py: used to update the database with the files stored in the source_docs folder. Add and delete files.
download_mistral.py : allows you to download the local mistral model and the associated tokenizer.
settings.py : contains setting for the chroma database
gpt_quest.py : sets up the RAG architecture
interface.py : streamlit code for the graphical interface


## Use via api

1. Creating or updating the DB 
Add and remove files from the ‘source_docs’ folder, then launch the vectorizer.py file.

2. Launch the chatbot 
Run the interface.py script with the command ‘streamlit run interface.py’.

## Local use

1. Creating or updating the DB 
Add and remove files from the ‘source_docs’ folder, then run the vectorizer.py file.

2. Download the Mistral model.
Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and accept the terms of use for the HuggingFace model.
Run the ‘download_mistral.py’ script.

3. Launching the chatbot 
Run the interface.py script with the ‘streamlit run interface.py’ command.



## Api pricing

API Pricing
                            Entree                      Sortie
mistral-embed	            0.1€ /1M tokens	            $0.1 /1M tokens
open-mistral-7b		        0.2€ /1M tokens	            0.2€ /1M tokens
GPT-4o mini 		        0.150€ /1M tokens	        0.6€ /1M tokens
text-embedding-3-small      0.02€ /1M tokens



## Local configuration 

Mistral 7b -> GPU with 24 Gb of VRam preferred (Example: Nvidia L4) or 16 Gb of VRam (Example: Nvidia T4)


## To go further

-> Quantization 
Quantization in the context of using Large Language Models (LLMs) refers to the process of reducing the precision of the numerical values used in the model's computations. We can go from 32 bits to 8 bits for example. It will make the model more efficient in terms of memory usage. 


-> Perform hybrid search 
An update of the project could be in the search. Hybrid search would permit to retrieve relevant chunks not only based on their meaning but also on relevant key words. That can be particularly useful when it comes to fields with specific terms. 