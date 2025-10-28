import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

env_path = Path(__file__).resolve().parent.parent / '.env' #find absolute path of ../
load_dotenv(dotenv_path=env_path)

pdf_folder = raw_documents_path
persist_directory = os.environ["VECTORDB_PATH"]
minor_models_device = os.environ["MINOR_MODELS_DEVICE"]
raw_documents_path = os.environ["RAW_DOCUMENTS_PATH"]

chunk_size_by_char = 1500    # ~400 tokens
chunk_overlap_by_char = 300  # ~80 tokens
        
embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
assert embeddings.embed_query("Hello")
print("Embeddings test OK.")

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
max_batch_size = 5461

if os.path.isdir(persist_directory):
    print("Found RAG library. Quit.")
else:
    print(f"Prepare RAG library to folder: {persist_directory}")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_by_char, chunk_overlap=chunk_overlap_by_char)
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}")
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            print("... Splitting")
            chunk = text_splitter.split_documents(docs)
            print(f"{len(chunk) // max_batch_size + 1} batches... Embedding & saving")
            for b in batch(chunk, max_batch_size):
                vector_store.add_documents(b)
            # vector_store.add_documents(chunk)
            print("... Done.")
    print("Library preparation finished.")
    
