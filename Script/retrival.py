## Importing Packages
import warnings
warnings.filterwarnings("ignore")
from langchain_chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
## Loading the Embedding Transformer

#embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")
embeddings = HuggingFaceEmbeddings(model_name="baconnier/Finance_embedding_large_en-V0.1")

## Creating the Prompt Template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

## Testing with a Sample Query
query = "What do you mean by Impact investing?"

## Loading the Vector Store
load_vector_store = Chroma(persist_directory="stores/4cosine", embedding_function=embeddings)

## Retrival of Chunks with k-setting-> 3
docs = load_vector_store.similarity_search_with_score(query=query, k=3)
docs = load_vector_store.similarity_search_with_score(query=query, k=3)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})