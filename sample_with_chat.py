#!/usr/bin/env python
# coding: utf-8

# In[2]:


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# In[3]:


import streamlit as st
st.title("HEDIS Chatbot")


# In[4]:


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# In[6]:


def get_documents_from_pdf(pdf_file):
    loader=PyPDFLoader(pdf_file)
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
    docs=text_splitter.split_documents(data)
    return docs


# In[7]:


def create_db(docs):
    vectorstore=Chroma.from_documents(documents=docs,embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key='AIzaSyBwQcJjZpsofcU5uegYRaUY-49doADgDhM'))
    return vectorstore


# In[8]:


def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",google_api_key='AIzaSyBwQcJjZpsofcU5uegYRaUY-49doADgDhM',temperature=0,max_tokens=None)
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use twenty sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
    )


    chain = create_stuff_documents_chain(llm,prompt
    )


    retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":20})
    retriever_chain=create_retrieval_chain(retriever,chain)
    return retriever_chain


# In[9]:


def process_chat(chain,question,chat_history):
    response=chain.invoke({
    "input": question,
    "chat_history":chat_history
    })
    return response["answer"]


# In[10]:


if __name__=='__main__':
    docs=get_documents_from_pdf("ABHKY Hedis Toolkit 2026.pdf")
    vectorstore=create_db(docs)
    #chain=create_chain(vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chain" not in st.session_state:
        st.session_state.chain = create_chain(vectorstore)






# In[ ]:





# In[11]:


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# In[12]:


query=st.chat_input("Say Something?")
if query:

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_chat(
                st.session_state.chain,
                query,
                st.session_state.chat_history
            )
            st.write(response)

    # Save History
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=response))


# In[ ]:




