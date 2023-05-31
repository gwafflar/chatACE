from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def design_gui() :
    st.set_page_config(page_title="Project INFO-H-512 : Current Trends of AI")
    st.header("Answer question from PDF using Large Language Models 💬")


def extract_text_from_pdf(pdf) :
    #read the pdf
    pdf_reader = PdfReader(pdf) #return the content of each page of the pdf
    text = ""
    for page in pdf_reader.pages: #get the content of all pages
        text += page.extract_text()
    return text 

@st.cache_data
def split_text_into_chunks(text) : 
    print("function split_text_into_chunks is called.")
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000, #len of a chunk
    chunk_overlap=200,  #chunks are overlapping (to avoid losing context)
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks 

@st.cache_data
def get_embeddings():
    embeddings = OpenAIEmbeddings()
    print("function get_embeddings is called.")
    return embeddings

@st.cache_data
def get_knowledge_base_from_chunks(chunks): #rename
    print("function get_knowledge_base_from_chunks is called.")
    embeddings = get_embeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base



def main():
    load_dotenv()
    
    design_gui()

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        chunks = split_text_into_chunks(text)
        knowledge_base = get_knowledge_base_from_chunks(chunks)
      
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        temperature = st.slider('Select temperature (randomness)', 0.0, 1.0)
        if user_question:
            #docs = knowledge_base.similarity_search(user_question)
            
            """llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
               
            st.write(response)"""
            st.write(user_question)
        else :
            st.write("There is no question for now")
    

if __name__ == '__main__':
    main()
 
