from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message as st_chat


import base64

from extract_and_clean_text import *
from language_and_ai import *


ACE_DIRECTORY = "data/ACE/"

def init_session_variables() :
    #init session variables
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'choice' not in st.session_state : 
        st.session_state['choice'] = "Null"
    if 'displayFile' not in st.session_state : 
        st.session_state['displayFile'] = "None"
    if 'knowledge_base' not in st.session_state : 
        st.session_state['knowledge_base'] = "None"
    if 'analyze_run_ACE' not in st.session_state : 
        st.session_state['analyze_run_ACE'] = False
    st.session_state['analyze_run_PDF'] = False
    if 'LM model' not in st.session_state : 
        st.session_state['LM model'] = "None"
    if 'Part2' not in st.session_state : 
        st.session_state['Part2'] = False
    if 'placeholder' not in st.session_state : 
        st.session_state['placeholder'] = ""
 
def design_gui() :
    st.set_page_config(page_title="ChatACE - Project INFO-H-512 : Current Trends of AI")
    st.header("Answer question from PDF using Large Language Models 💬")

def provide_chunks_and_generate_answer(knowledge_base, user_question) :
    #get chunks and call model functions
    with st.expander('Pertinent chunks of the PDFs') :
        docs = knowledge_base.similarity_search(user_question)
        st.write(docs)
    with st.spinner("Generating answer...") :
        if st.session_state["LM model"] == "GPT4" :#check model name :
            response = generate_answer_from_OpenAI(docs, user_question)
        elif st.session_state["LM model"] == "Bloom" :
            response = generate_answer_from_bloom(docs, user_question)
        elif st.session_state["LM model"] == "Vicuna" :
            response = generate_answer_from_vicuna(docs, user_question)
        else :
            st.error("Please select a LM model") #btw LM model ne se dit pas 
            response = "Error : no Language model selected."
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
    #display the list of all the files of the ACE in an expander 
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


def get_placeholder() :
    #pre-print a string in the input text
    placeholder = ""
    if st.session_state['choice'] == "ACE" :
        placeholder = "Quel est le rôle de l'Association des Cercles Etudiants ?"
    elif st.session_state['choice'] == "newPDF" :
        placeholder = "What is the topic of the document ?"
    else :
        placeholder="error"    
        print("Error : no user choice")
    return placeholder


def display_option_ACE():
    #the elements of the page one the "ACE option" is selected
    st.write(":two: You can click to display all the rules of the Association des Cercles Etudiants. You can display or download any of these files.")
    display_ACE_files()
    if st.session_state['displayFile'] != "None" :
        hide_file = st.button("Hide document")
        displayPDF(st.session_state['displayFile'])
        if hide_file :
            st.session_state['displayFile'] = "None"
    st.write(":three: Click on the button to parse and analyze all the files. This way, the content of the documents will be used by the language model to answer questions about it.")
    run_analyze = st.button("Analyze files")
    if run_analyze :
        st.session_state['analyze_run_ACE'] = True
    if st.session_state['analyze_run_ACE'] == True :
        try : 
            analyze_text_ACE()  
            st.info("The documents are now divided into chunks and can be used by language models.")
            st.write(":four: You can now ask any question about the rules of the Association des Cercles Etudiants ⬇️")
            st.session_state["Part2"] = True
        except : 
            st.error("Error while analyzing the file.")


def display_pdf_option() :
    pdfFile = st.file_uploader(":two: Upload your PDF", type="pdf")
    run_analyze = False
    if pdfFile is not None:
        st.write(":three: Click on the button to parse and analyze the PDF. This way, the content of the document will be used by the language model to answer questions about it.")
        run_analyze = st.button("Analyze file")
    else :
        st.session_state['Part2'] = False
    if run_analyze :
        st.session_state['analyze_run_PDF'] = True
    if st.session_state['analyze_run_PDF'] == True :
        try :
            analyze_text_newPDF(pdfFile)
            st.session_state["Part2"] = True  
            st.info("The document is now divided into chunks and can be used by language models.")
            st.write(":four: You can now ask any question about your PDF ⬇️")
        except : 
            st.error("Error while analysing the PDF file. Please be sure the file contains text.")

def main():
    load_dotenv()
    init_session_variables()
    design_gui()
    print(os.getenv("OPENAI_API_KEY"))
    st.write(os.getenv("OPENAI_API_KEY"))
    print(OPENAI_API_KEY)
    st.write(OPENAI_API_KEY)
    print(API_KEY_HuggingFace)
    st.write(API_KEY_HuggingFace)

    user_choice = st.selectbox(
            "Select what document you want the model to answer questions about",
            ("Select an option", "Statuts et règlements de l'ACE", "Upload a new document"))
    st.write(":one: Select an option in the list")

    # Select between ACE and new PDF
    if user_choice == "Select an option" :
        st.session_state['choice'] = "Null"
        st.session_state['Part2'] = False
    elif user_choice == "Statuts et règlements de l'ACE" :
        st.session_state['choice'] = "ACE"
        display_option_ACE()
    elif user_choice == "Upload a new document" :
        st.session_state['choice'] = "newPDF"
        display_pdf_option()
    else : 
        st.session_state['choice'] = "Null"
        st.write("Error : unknow choice")


    # Once the file(s) is/are analysed, display the rest of the components
    if st.session_state["Part2"] == True :
        placeholder = get_placeholder()
        user_question = st.text_input("Ask a question:", placeholder=placeholder)  
        #temperature = st.slider('Select temperature (randomness)', 0.0, 1.0)
        model = st.radio(":five: Select a language model :", 
                    key="LM model", 
                    options=["GPT4", "Bloom", "Vicuna"],
                 )       
        st.write(":six: Click on the button :arrow_down: to get an answer !")
        run_query = st.button("Answer me")
        reset_chat_button = st.button("🔄 Reset history chat")

        #Display the chat with question and generated answer
        if user_question and run_query:
            st.session_state['chat_history'].append((user_question, True))
            display_chat_history(st.session_state['chat_history'])
            print("\tQuestion : ", user_question)
            response="ok"
            if st.session_state['knowledge_base'] != "None" :
                #call function to generate answer with model
                response = provide_chunks_and_generate_answer(st.session_state['knowledge_base'], user_question)
                pass
            else :
                st.error('No knowledge_base yet ! Please click on "Analyze files" first.')
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

 