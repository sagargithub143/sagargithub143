#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.tools.tavily_search import TavilySearchResults



from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


# In[2]:


st.title("HEDIS Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# In[ ]:


def load_pdf(pdf):

    loader = PyPDFLoader(pdf)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)


# In[ ]:


def create_vectorstore(docs):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key="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")

    vectorstore = Chroma.from_documents(documents=docs,embedding=embeddings)

    return vectorstore


# In[ ]:


def create_chain(vectorstore):

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key="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"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer only from the supplied context.
If the answer is not available in the context, say:

NOT_FOUND_IN_DOCS

Context:
{context}
"""
            ),
            ("human", "{input}")
        ]
    )

    combine_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":20}
    )

    return create_retrieval_chain(
        retriever,
        combine_chain
    )


# In[ ]:



os.environ["TAVILY_API_KEY"]="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"
web_search = TavilySearchResults(max_results=3)



# In[ ]:


def ask_question(question):

    response = st.session_state.chain.invoke(
        {"input": question}
    )

    answer = response["answer"]

    if "NOT_FOUND_IN_DOCS" not in answer:

        return answer

    results = web_search.invoke(question)

    context = "\n".join(
        r["content"] for r in results
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key="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"
    )

    prompt = f"""
The internal PDF did not contain the answer. If the answer not found in the give file provide details to user that seraching in online

Use the following web search results.

{context}

Question:
{question}
"""

    return llm.invoke(prompt).content


# In[ ]:


if "chain" not in st.session_state:

    docs = load_pdf("ABHKY Hedis Toolkit 2026.pdf")

    vectorstore = create_vectorstore(docs)

    st.session_state.chain = create_chain(vectorstore)


# In[ ]:


for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    else:
        with st.chat_message("assistant"):
            st.write(message.content)


# In[ ]:


question = st.chat_input("Ask a question")

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask_question(question)

            st.write(answer)

    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

