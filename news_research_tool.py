import os
import streamlit as st
import pickle
import time
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

file_path = "faiss_index.pkl"

#model_name = sys.argv[1]
model_name = 'LLaMA3-70b'

def news_data(): 
    urls=[]
    for i in range(3):
        url = st.sidebar.text_input(f'URL {i+1}')
        urls.append(url)
    return urls

def load_llm(model_name):
    if model_name == 'gemini-pro':
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    elif model_name == 'LLaMA3-8b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    elif model_name == 'LLaMA3-70b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    elif model_name == 'LLaMA2-70b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama2-70b-4096")
    elif model_name == 'Mixtral-8x7b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    elif model_name == 'Gemma-7b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="gemma-7b-it")
    return model

def load_url_data(urls,main_placeholder):
    loader = UnstructuredURLLoader(urls = urls)
    #print('urls', urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    #print('data', data)
    return data


def check_url_access(data,main_placeholder):
    if any('Access Denied' in doc.page_content for doc in data):
        urls_with_access_denied = [doc.metadata.get('source', '') for doc in data if 'Access Denied' in doc.page_content]
        access_denied_msg = "Sorry, I don't have permission to access the following links:\n\n"
        for url in urls_with_access_denied:
            access_denied_msg += f"- {url}\n"
        main_placeholder.text(access_denied_msg)
        stop_processing = True  # Set flag to stop further processing
        #print(stop_processing)

def create_chunks_and_embeddings(data,main_placeholder):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    #print('docs',docs)
    # Create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    #vectorstore.save_local("faiss_index")
    main_placeholder.text("Embedding Vector Started Building...✅✅✅") #st.text()
    time.sleep(2)
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

def create_prompt_template(llm,vectorstore):
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $200 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


