#####################################################################################################################################################################
################## This script reads documents from the source directory to create and update a database.############################################################
#####################################################################################################################################################################
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os 
import glob
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
from multiprocessing import Pool
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import CHROMA_SETTINGS
import chromadb
import os



if not load_dotenv():
    print(".env file misssing or empty.")


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = int(os.environ.get('CHUNK_SIZE'))
chunk_overlap = int(os.environ.get('CHUNK_OVERLAP'))
batch_size = int(os.environ.get('BATCH_SIZE'))


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}) 
}


def load_single_document(file_path: str) -> List[Document]:
    """**Load a single document from a file path**

    Args:
        file_path (str): path document file

    Raises:
        ValueError: unsupported file extension

    Returns:
        List[Document]: list of Document objects
    """
    #check the extension of the file
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    # loader mapping
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args) # load the document with additional arguments args
        return loader.load() #Load data into Document objects. List[Document]

    raise ValueError(f"Unsupported file extension '{ext}'")



def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """**Load all documents from a source directory**

    Args:
        source_dir (str): directory containing the source documents
        ignored_files (List[str], optional): files to ignre when creating the db. Defaults to [].

    Returns:
        List[Document]: list of different chunks of documents
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True) 
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """**Load documents and split in chunks**

    Args:
        ignored_files (List[str], optional): files to ignore when creating the database. Defaults to [].

    Returns:
        List[Document]: list of different chunks of documents
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files) 
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str, embeddings: OpenAIEmbeddings, chroma_client) -> bool:
    """returns whether or not the vectorstore exists in the specified directory

    Args:
        persist_directory (str): directory of the vectorstore
        embeddings (): embedding model
        chroma_client (): Chroma database client

    Returns:
        bool: Existence of the vectorstore
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)#Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True




def sources_remover(source_dir: str, collection: dict):
    """returns ids of vectors to delete from the db

    Args:
        source_dir (str): directory containing the source documents
        collection (dict): vectore db informations

    Returns:
        list: list of all vectors to remove from the db
    """
    
    #identify sources in the db
    data = collection['metadatas']
    db_sources = {item['source'] for item in data} #distinct sources in the db

    #identify sources in the source directory
    directory_sources = []
    for ext in LOADER_MAPPING:
        directory_sources.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True) # lower case, consider all files in subdirectories
        )

    sources_to_delete = list(db_sources - set(directory_sources))

    ids_to_del = []
    for idx, metadata in enumerate(collection['metadatas']) : 
        if metadata['source'] in sources_to_delete:
            ids_to_del.append(collection['ids'][idx])
        

    return ids_to_del, sources_to_delete

def main():


    match embeddings_model_name:
        case "OpenAI api":
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        case "Mistral api":
            # Define the embedding model
            embeddings = MistralAIEmbeddings(model="mistral-embed")
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Embedding model {embeddings_model_name} is not supported. Please choose one of the following: Open AI api, Mistral local, Mistral api")

    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    
    #add the embedding inside the db when is exists. otherwise, it creates the db
    if does_vectorstore_exist(persist_directory, embeddings, chroma_client):
        # Update and store locally vectorstore
        print(f"Updating existing vectorstore in {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        #remove sources no longer in the source directory
        ids_to_delete, sources_to_delete = sources_remover(source_directory, collection= collection)
        if ids_to_delete : 
            db._collection.delete(ids_to_delete)
            print(str(len(sources_to_delete)) + " sources removed from the db : " + str(sources_to_delete))
        #add new docs
        ignored_files = list(set([metadata['source'] for metadata in collection['metadatas']])) 
        documents = process_documents(ignored_files= ignored_files) #ignore all source already used
        print(f"Creating embeddings. May take some minutes...")

        # Iterate through the documents in batches
        for i in range(0, len(documents), batch_size):
            # Get the current batch
            batch = documents[i:i + batch_size]
            # Add the current batch to the database
            db.add_documents(batch)

    else:
        #Create and store locally vectorstore
        print("Creating new vectorstore")
        documents = process_documents() #process all documents
        the_texts = [doc.page_content for doc in documents]
        print(f"Creating embeddings. May take some minutes...")
        
        # Split the first batch
        first_batch = documents[:batch_size]

        # Initialize the database with the first batch
        db = Chroma.from_documents(first_batch, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
        if len(documents) > batch_size:
            
            # Iteratively add the remaining documents in batches
            for i in range(batch_size, len(documents), batch_size):
                # Get the current batch
                batch = documents[i:i + batch_size]
                
                # Add the current batch to the database
                db.add_documents(batch)
        db = None
   


if __name__ == "__main__":
    main()