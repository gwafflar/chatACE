from PyPDF2 import PdfReader
from re import sub
from typing import Callable, List
import streamlit as st
import os
from language_and_ai import *

ACE_DIRECTORY = "data/ACE/"

def merge_hyphenated_words(text: str) -> str:
    return sub(r"(\w)-\n(\w)", r"\1\2", text)

def fix_newlines(text: str) -> str:
    return sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_multiple_newlines(text: str) -> str:
    return sub(r"\n\s*\n", "\n", text)

def remove_foot_page(text: str) -> str:
    pattern = r"Association\s*des\s*Cercles\s*Étudiants\s*de\s*l’ULB\s*–\s*ASBL\s*N°\s*d’entreprise\s*:\s*414\.410\.031\s*\(\s*RPM\s*Tribunal\s*de\s*l’entreprise\s*Francophone\s*de\s*Bruxelles\s*\)\s*Siège\s*:\s*Avenue\s*Paul\s*Héger,\s*22\s*\(\s*CP\s*166\/09\s*\)\s*–\s*1000\s*Bruxelles\s*\(\s*BE\s*\)\s*Tél\.\s*:\s*02\s*650\s*25\s*14\s*–\s*E-mail\s*:\s*bureau\s*@ace\s*-ulb\.be\s*\d*"
    return sub(pattern, " ", text)

def remove_header_page(text: str) -> str:
    pattern = r"Association des Cercles Étudiants de l’UL\s*B – (Statuts|Règlement d’Ordre Intérieur) \s*\(\d+ mai 2022\s*\)"
    return sub(pattern, " ", text)

def remove_points_in_table_of_contents(text: str) -> str:
    return sub(r"\.{4,}", "", text)

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
    except Exception as e : 
        print(e)
        st.error("Cannot communicate with LLM.")
        st.error(e)

@st.cache_data
def analyze_text_newPDF(pdf) :
    text = extract_text_from_one_pdf(pdf)
    text = clean_text(text, [merge_hyphenated_words, remove_multiple_newlines]) #+fix_newlines?
    chunks = split_text_into_chunks(text)
    try : 
        st.session_state['knowledge_base'] = get_knowledge_base_from_chunks(chunks)
    except : 
        st.error("Cannot communicate with LLM")
