import os
import warnings
warnings.filterwarnings("ignore")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

try:
    embeddings = HuggingFaceEmbeddings(model_name="baconnier/Finance_embedding_large_en-V0.1")
    print("Embeddings loaded.....")
except:
    print("Error loading the embeddings...")

try:
    loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Document Loaded...")
except:
    print("error in loading the documents...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=722, chunk_overlap=80)
texts = text_splitter.split_documents(documents)


try:
    vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/4cosine")
    print("vector store created...")
except:
    print('Error in storing vectors store')