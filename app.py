"""
This code was created by yembot
"""

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import os

# load_dotenv()
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDHinuKQkQrKhYPRFrTWXDykmidgclwdtk'

# vector = embeddings.embed_query("hello, world!")

llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key='AIzaSyDHinuKQkQrKhYPRFrTWXDykmidgclwdtk')

st.set_page_config(page_title="ASK YOUR PDF")

with st.sidebar:
    st.title('CHATWITHPDF')
    st.markdown('''
    
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [Gemini]('https://ai.google.dev/') LLM model            
                
                
                ''')
    add_vertical_space(5)
    # st.write('made with ')

def main():
    
    st.header('ASK YOUR PDF')

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf:
        st.write(pdf.name)

        if pdf is not None:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                separators="/n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        chunks = text_splitter.split_text(text=text)

            # st.write()

    
        # store_name = pdf.name[:-4]

        # if os.path.exists(f"{store_name}.pk1"):
        #     with open(f"{store_name}.pk1", "rb") as f:
        #         Vectorstore = pickle.load(f)
        #     st.write('Embeddings Loaded from the Disk')
        # else:
        #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        #     Vectorstore = embeddings.aembed_documents(chunks)
        #     # Vectorstore = FAISS.from_texts(chunks, embedding=embeddings )
        #     with open(f"{store_name}.pk1", "wb") as f:
        #         pickle.dump(Vectorstore, f)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        knowledge_base = FAISS.from_texts(chunks, embedding=embeddings )

        user_question = st.text_input("Ask questions about your PDF")
        # st.write(user_question)

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key='AIzaSyDHinuKQkQrKhYPRFrTWXDykmidgclwdtk')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                 user_input = f'''
                    System prompt:
                    you are an Ai created by Rufai Sodiq Sijuade from community innovation hub alimosho. your name is CHATWITHPDF. you are designed to help users summarize their PDF and also allow them interact with it, when a user asks you a question you are to give response based on the content of the PDF. if the user's question doesn't correlate with the content of the PDF. you are not allowed to perform any other thing for user except from allowing users interact with you and also summarizing PDF. you are not allowed to write code for any user. when a user ask you to write code you are to reply saying "sorry I am not designed to write code"

                    user input:
                    {user_question}
                '''
                 response = chain.run(input_documents=docs, question=user_input)
                 print(cb)
           
            st.write(response)

               


main()