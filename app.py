## Importing Packages
import os
import time
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings 
import streamlit as st 

## Settings-Configuration-> HuggingFace
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

## UI Configuration
st.set_page_config(page_title="RAG Implementation - Asset Management  Domain Data")
st.title("Asset Management Retrieval Augmented Generation Implementation")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## Prompt Template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

## Laoding the Transformers
#embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")
embeddings = HuggingFaceEmbeddings(model_name="baconnier/Finance_embedding_large_en-V0.1")

## Loading the VectorStore
load_vector_store = Chroma(persist_directory="stores/4cosine", embedding_function=embeddings)

## Defining retriever with top -> 2
retriever = load_vector_store.as_retriever(search_kwargs={"k":2})

## Loading the OpenSource LLM's
repo_id = "llmware/bling-sheared-llama-1.3b-0.1"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_length": 500}
)

chain_type_kwargs = {"prompt": prompt}

## Defining Q&A Chain for the LLMs
def qa_chain():
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    #verbose=True
    )
    return qa

qa = qa_chain()

def main():
    """
    Main function to handle user input and generate assistant responses.

    Checks for user input using Streamlit chat_input, adds user messages to chat history,
    displays user messages in the chat message container, calls the Q&A function to get
    the assistant's response, and displays the assistant's response with a simulated typing effect.
    """
    if prompt_ms := st.chat_input("Fin-Tech Asset Management Bot Here..."):
        ## Adding user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_ms})
        ## Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt_ms)

        ## Call QA function to get assistant response
        text_query = prompt_ms
        text_response = qa(text_query)['result']

        ## Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            ## Simulate stream of response with milliseconds delay
            for chunk in text_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                ## Add a blinking cursor to simulate typing in the UI
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        ## Add assistant response to chat history in the Streamlit UI
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()


