from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
import requests #for Bloom API
import replicate #for Vicuna API
import os

API_KEY_HuggingFace = os.getenv('API_KEY_HuggingFace')
API_URL_HuggingFace = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": f"Bearer {API_KEY_HuggingFace}"}
VICUNA_MODEL = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"

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
    separator = "."
    if st.session_state['choice'] == "ACE" :
        separator = "Art." #for ACE file, separate per article
    elif st.session_state['choice'] == "newPDF" :
        separator = "\n"
    text_splitter = CharacterTextSplitter(
    separator=separator,
    chunk_size=1000, #len of a chunk
    chunk_overlap=200,  #chunks are overlapping (to avoid losing context)
    length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks 


def extract_text_from_chunks(chunks) :
    text = ""
    for chunk in chunks :
        text+=chunk.page_content+"\n"
    return text

# ========== OpenAI ============ #

def reformulate_price_request(cb) :
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

prompt_bloom = ["""Dialogue entre un Alan et Marie. Marie a accès aux informations suivantes dans le réglement: 
        Début des informations : """,
        """
        - fin des informations à disposition de Marie.
        Le Alan pose la question : '""",
        " ?' Marie lui répond naturellement "]
#TODO : find better prompt example on the internet (eg on the vicogne github page)
def create_bloom_prompt(chunks, user_question) :
    raw_text = extract_text_from_chunks(chunks[:2])
    #prompt = prompt_bloom[0] + raw_text + prompt_bloom[1] + user_question + prompt_bloom[1]
    #prompt="Tu es mon avocat. Je te fournis un texte de réglement. Sur base de ce texte, réponds à ma question que je te poserai après. Voilà mon texte : \n" + raw_text + "\n\n\n Sur base de ces informations, réponds à cette question, en citant les articles dont il relève : " + user_question
    prompt = raw_text + "\nQuestion: " + user_question + "\nRéponse: "
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
    

# =========== Vicuna =================== #

prompt_vicuna = ["Tu es mon avocat. Je te fournis un texte de réglement. Sur base de ce texte, réponds à ma question que je te poserai après. Voilà mon texte : \n",
                "\n\n\n Sur base de ces informations, réponds à cette question, en citant les articles dont il relève : "]
def create_vicuna_prompt(chunks, user_question) :
    raw_text = extract_text_from_chunks(chunks[:1])
    #prompt = prompt_vicuna[0] + raw_text + prompt_vicuna[1] + user_question
    #prompt = "Read the following infor : \n" + raw_text + "\n Sur base de ces informations, dites moi " + user_question.lower()
    prompt =  "\n\n Voici un texte. Sur base de ce texte, réponds à la question.\n " + raw_text + "\n\nQuestion : " + user_question + "\n Réponse: "
    print(prompt)
    return prompt

def generate_answer_from_vicuna(chunks, user_question) :
    prompt = create_vicuna_prompt(chunks, user_question) 
    answer = ""
    output = replicate.run(
        VICUNA_MODEL,
        input={"prompt": prompt, "max_lenght":len(prompt)*3+300}
    )
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        answer += item
        print(item, end=" ")
    print("-end of answer-")
    return answer