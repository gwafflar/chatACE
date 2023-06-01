from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message as st_chat
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import base64
import os
import re 
from typing import Callable, List
ACE_DIRECTORY = "data/ACE/"
#TODO : 
    #Tutorial in sidebar + explaination page
    #ACE : hide PDF once displayed
    #use the temperature
    #insert other parameter (top_k ?)
    #use other LL models ? 
    #rename knowledge database
    #add metadata in every chunk (doc + article n°)
    #change chat icon
    #error : header du ROI apparait à nouveau -> lier au fait qu'on ai laissé les \n ? 
    #ajouter les chunks pertinents dans un collapse
    #refactoring en plusieurs fichiers

def design_gui() :
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'choice' not in st.session_state : #deprecated ? 
        st.session_state['choice'] = "Null"
    if 'displayFile' not in st.session_state : 
        st.session_state['displayFile'] = "None"
    if 'knowledge_base' not in st.session_state : 
        st.session_state['knowledge_base'] = "None"

    st.set_page_config(page_title="Project INFO-H-512 : Current Trends of AI")
    st.header("Answer question from PDF using Large Language Models 💬")
    st.sidebar.markdown("""# Home
                        Insert tutorial here""")


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n\s*\n", "\n", text)

def remove_foot_page(text: str) -> str:
    pattern = r"Association\s*des\s*Cercles\s*Étudiants\s*de\s*l’ULB\s*–\s*ASBL\s*N°\s*d’entreprise\s*:\s*414\.410\.031\s*\(\s*RPM\s*Tribunal\s*de\s*l’entreprise\s*Francophone\s*de\s*Bruxelles\s*\)\s*Siège\s*:\s*Avenue\s*Paul\s*Héger,\s*22\s*\(\s*CP\s*166\/09\s*\)\s*–\s*1000\s*Bruxelles\s*\(\s*BE\s*\)\s*Tél\.\s*:\s*02\s*650\s*25\s*14\s*–\s*E-mail\s*:\s*bureau\s*@ace\s*-ulb\.be\s*\d*"
    return re.sub(pattern, " ", text)

def remove_header_page(text: str) -> str:
    pattern = r"Association des Cercles Étudiants de l’UL\s*B – (Statuts|Règlement d’Ordre Intérieur) \s*\(\d+ mai 2022\s*\)"
    return re.sub(pattern, " ", text)

def remove_points_in_table_of_contents(text: str) -> str:
    return re.sub(r"\.{4,}", "", text)

@st.cache_data
def clean_text(text : str, _cleaning_functions: List[Callable[[str], str]]) :
    for cleaning_function in _cleaning_functions:
        text = cleaning_function(text)
    return text

def extract_text_from_one_pdf(pdf) :
    #read the pdf
    pdf_reader = PdfReader(pdf) #return the content of each page of the pdf
    text = ""
    for page in pdf_reader.pages: #get the content of all pages
        text += page.extract_text()
    return text 

@st.cache_data
def extract_text_from_multiple_pdf() :
    text = ''
    for filename in os.listdir(ACE_DIRECTORY):
        if filename.endswith(".pdf") :
            text += extract_text_from_one_pdf(ACE_DIRECTORY+filename)
    return text 

@st.cache_data
def split_text_into_chunks(text) : 
    print("function split_text_into_chunks is called.")
    #/!\ separator different for ACE and newPDF
    text_splitter = CharacterTextSplitter(
    separator="Art.",
    chunk_size=1000, #len of a chunk
    chunk_overlap=200,  #chunks are overlapping (to avoid losing context)
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks 

def reformulate_price_request(cb) :
    """txt = cb.split("\n")
    token_used = int(txt[0].split(": ")[1])
    token_prompt = int(txt[1].split(": ")[1])
    token_completion = int(txt[2].split(": ")[1])
    success = int(txt[3].split(": ")[1])
    cost = int(txt[4].split("$")[1])"""
    #return f"Tokens used : {token_prompt}+{token_completion}={token_used} ({token_prompt+token_completion}). Cost : ${cost}. - {success}."
    return f"Tokens used : {cb.prompt_tokens}+{cb.completion_tokens}={cb.total_tokens} ({cb.prompt_tokens+cb.completion_tokens}). Cost : ${cb.total_cost}."

@st.cache_data
def get_embeddings():
    print("function get_embeddings is called.")
    try :
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings()
            print(reformulate_price_request(cb), " ")
    except : 
        st.error("Error from OpenAI. Missing API KEY ? Or just callback function ? ")
        print("Error : missing API key ? ")
    return embeddings

@st.cache_data
def get_knowledge_base_from_chunks(chunks): #rename
    print("function get_knowledge_base_from_chunks is called.")
    try :
        embeddings = get_embeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base
    except:
        raise ErrorLLM

def generate_answer(knowledge_base, user_question) :
    print("generate_answer")
    with st.spinner("Generating answer...") :
        docs = knowledge_base.similarity_search(user_question)
        print(docs)
        with get_openai_callback() as cb:
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            print(reformulate_price_request(cb))
        #response = "ok"
    return response

def display_chat_history(chat_history) :
    for key, msg in enumerate(chat_history) :
        st_chat(msg[0], is_user=msg[1], key=str(200+key))

def displayPDF(file):
    # Opening file from file path
    file_path = ACE_DIRECTORY + file
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)   

def display_ACE_files() : 
    list_buttons = []
    with st.expander("List of source files (click to expand):", expanded=True) :
        for i, filename in enumerate(os.listdir(ACE_DIRECTORY)):
            if filename.endswith(".pdf") :
                col1, col2, col3 = st.columns([3,1,1])
                with col1:
                    st.write("📄 " + filename)
                with col2:
                    button_display = st.button("Display", key=i)
                    if button_display :
                        st.session_state['displayFile'] = filename
                with col3:
                    st.download_button(label="Download", key=100+i, data=filename, file_name=filename)

@st.cache_data
def analyze_text_ACE() :
    text = extract_text_from_multiple_pdf()
    cleaning_functions = [
        merge_hyphenated_words, remove_multiple_newlines, #fix_newlines ?
        remove_foot_page, remove_header_page, remove_points_in_table_of_contents
        ]
    text = clean_text(text, cleaning_functions)
    chunks = split_text_into_chunks(text)
    try : 
        st.session_state['knowledge_base'] = get_knowledge_base_from_chunks(chunks)
    except : 
        st.error("Cannot communicate with LLM")

@st.cache_data
def analyze_text_newPDF(pdf) :
    print("function analyze_text_ACE")
    text = extract_text_from_one_pdf(pdf)
    text = clean_text(text, [merge_hyphenated_words, remove_multiple_newlines]) #+fix_newlines?
    chunks = split_text_into_chunks(text)
    try : 
        st.session_state['knowledge_base'] = get_knowledge_base_from_chunks(chunks)
    except : 
        st.error("Cannot communicate with LLM")

def main():
    load_dotenv()
    
    design_gui()

    user_choice = st.selectbox(
            "Select what document you want the model to answer questions about",
            ("Select an option", "Statuts ACE", "Upload a new document", "other"))

    if user_choice == "Select an option" :
        #display welcome information (basically : select in the list what you want)
        st.write("Select an option in the list")
        st.session_state['choice'] = "Null"

    elif user_choice == "Statuts ACE" :
        st.session_state['choice'] = "ACE"
        display_ACE_files()
        if st.session_state['displayFile'] != "None" :
            displayPDF(st.session_state['displayFile'])
        analyze_text_ACE()        

    elif user_choice == "Upload a new document" :
        st.session_state['choice'] = "newPDF"
        st.info("/!\\ change function split_text_into_chunks")
        pdfFile = st.file_uploader("Upload your PDF", type="pdf")
        if pdfFile is not None:
            analyze_text_newPDF(pdfFile)

    else : 
        st.session_state['choice'] = "Null"
        st.write("To Do")

    if st.session_state['choice'] != "Null" :
        user_question = st.text_input("Ask a question about your PDF:", placeholder="Quel est le rôle de l'Association des Cercles Etudiants ?") #change label and add transparent proposition ? 
        temperature = st.slider('Select temperature (randomness)', 0.0, 1.0) #default value ? 
        reset_chat_button = st.button("🔄 Reset history chat")
        if user_question:
            st.session_state['chat_history'].append((user_question, True))
            display_chat_history(st.session_state['chat_history'])
            print("\tQuestion : ", user_question)
            response="ok"
            if st.session_state['knowledge_base'] != "None" :
                #response = generate_answer(st.session_state['knowledge_base'], user_question)
                pass
            else :
                st.error("No knowledge_base yet !")
            st_chat(response)
            print("\tAnswer : ", response)
            st.session_state['chat_history'].append((response, False))
        else :
            st.write("There is no question for now")

        if reset_chat_button :
            st.session_state['chat_history'] = []
            display_chat_history(st.session_state['chat_history'])
if __name__ == '__main__':
    main()

 