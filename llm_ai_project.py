from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message as st_chat
from langchain.text_splitter import CharacterTextSplitter


import base64

from extract_and_clean_text import *
from language_and_ai import *


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
    if 'analyze_run_ACE' not in st.session_state : 
        st.session_state['analyze_run_ACE'] = False

    st.set_page_config(page_title="Project INFO-H-512 : Current Trends of AI")
    st.header("Answer question from PDF using Large Language Models 💬")
    st.sidebar.markdown("""# Home
                        Insert tutorial here""")




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

def provide_chunks_and_generate_answer(knowledge_base, user_question) :
    print("generate_answer")
    with st.expander('Pertinent chunks of the PDFs') :
        docs = knowledge_base.similarity_search(user_question)
        st.write(docs)
    with st.spinner("Generating answer...") :
        response = generate_answer_from_OpenAI(docs, user_question)
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
    with st.expander("List of source files (click to expand):", expanded=False) :
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
        st.write("You can click to display all the rules of the Association des Cercles Etudiants. You can display or download any of these files.")
        display_ACE_files()
        if st.session_state['displayFile'] != "None" :
            displayPDF(st.session_state['displayFile'])
            #create button to hide the file (using st.empty() ?)
        st.write("Click on the button to parse and analyze all the files. This way, the content of the documents will be used by the language model to answer questions about it.")
        run_analyze = st.button("Analyze files")
        if run_analyze :
            st.session_state['analyze_run_ACE'] = True
        if st.session_state['analyze_run_ACE'] == True :
            analyze_text_ACE()  
            st.info("The documents are now divided into chunks that will be sent to the language model.")
            st.write("You can now ask any question about the rules of the Association des Cercles Etudiants ⬇️")
      

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
        user_question = st.text_input("Ask a question:", placeholder="Quel est le rôle de l'Association des Cercles Etudiants ?") #change label and add transparent proposition ? 
        #temperature = st.slider('Select temperature (randomness)', 0.0, 1.0) #default value ? 
        run_query = st.button("Answer me")
        reset_chat_button = st.button("🔄 Reset history chat")
        if user_question and run_query:
            st.session_state['chat_history'].append((user_question, True))
            display_chat_history(st.session_state['chat_history'])
            print("\tQuestion : ", user_question)
            response="ok"
            if st.session_state['knowledge_base'] != "None" :
                response = provide_chunks_and_generate_answer(st.session_state['knowledge_base'], user_question)
                pass
            else :
                st.error("No knowledge_base yet !")
            st_chat(response)
            print("\tAnswer : ", response)
            st.session_state['chat_history'].append((response, False))
        else :
            st.write("Please write your question.")

        if reset_chat_button :
            st.session_state['chat_history'] = []
            display_chat_history(st.session_state['chat_history'])
if __name__ == '__main__':
    main()

 