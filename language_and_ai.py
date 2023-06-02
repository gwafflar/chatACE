from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
import requests
import os

API_KEY_HuggingFace = os.getenv('API_KEY_HuggingFace')
API_URL_HuggingFace = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": f"Bearer {API_KEY_HuggingFace}"}


# ====== Use embeddings to create knoledge base ======== #

@st.cache_data
def get_embeddings():
    print("function get_embeddings is called.")
    try :
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings()
            print(reformulate_price_request(cb), " ")
    except : 
        st.error("Error from OpenAI. Missing API KEY ?")
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

# ========== OpenAI ============ #

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
def generate_answer_from_OpenAI(_docs, user_question) :
    with get_openai_callback() as cb:
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=_docs, question=user_question)
        print(reformulate_price_request(cb))
    return response



# ========== Bloom ============ #

def extract_text_from_chunks(chunks) :
    text = ""
    for chunk in chunks :
        text+=chunk.page_content+"\n"
    return text

def create_bloom_prompt(chunks, user_question) :
    raw_text = extract_text_from_chunks(chunks)
    prompt = """Dialogue entre un demandeur, et un répondeur qui a accès aux informations suivantes dans le réglement: 
-- Début des informations dans le réglement à disposition du répondeur :""" + raw_text + """
-- fin des informations à disposition du répondeur
Le demandeur pose la question : '""" + user_question + " ?' Le répondeur lui répond naturellement "
    return prompt

def generate_answer_from_bloom(chunks, user_question) :
    prompt = create_bloom_prompt(chunks, user_question)
    payload = {
        "inputs": prompt,
         "parameters": {"max_new_tokens": 100,
                    "return_full_text": False}
        }
    output = query(payload)
    print(output)
    return output

def query(payload):
    response = requests.post(API_URL_HuggingFace, headers=headers, json=payload)
    return response.json()[0]['generated_text']
    