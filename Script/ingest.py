import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="baconnier/Finance_embedding_large_en-V0.1")

try:
    loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    print("Document Loaded...")
except:
    print("error in loading the documents...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=722, chunk_overlap=80)
texts = text_splitter.split_documents(documents)


vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/4cosine")
