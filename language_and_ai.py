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

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

API_KEY_HuggingFace = os.getenv('API_KEY_HuggingFace')

API_URL_HuggingFace = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": f"Bearer {API_KEY_HuggingFace}"}
VICUNA_MODEL = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"

def log_activity(model, user_question, response) :
    try :
        log_filename = "logs/"+model+".log"
        with open(log_filename, 'a') as f:
            f.write("Question :" + user_question + "\nRéponse : "+ response + "\n\n")
    except Exception as e: 
            print("Error while logging : ", e, "\n", model, user_question, response)

def print_logs() :
    models=["gpt4", "bloom", "vicuna"]
    for m in models :
        log_filename = "logs/"+m+".log"
        with open(log_filename, 'r') as f :
            print(f.read())
# ====== Use embeddings to create knowledge base ======== #

@st.cache_data
def get_embeddings():
    embeddings = ""
    try :
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings()
        print(reformulate_price_request(cb), " ")
    except Exception as e: 
        st.error("Error from OpenAI. Missing API KEY ?")
        st.error(e)
        print("Error : missing API key ? ")
    finally :
        return embeddings

@st.cache_data
def get_knowledge_base_from_chunks(chunks): #rename
    print("function get_knowledge_base_from_chunks is called.")
    try :
        embeddings = get_embeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base
    except Exception as e:
        st.error(e)


@st.cache_data
def split_text_into_chunks(text) : 
    print("function split_text_into_chunks is called.")
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
        log_activity("gpt4", user_question, response)
    return response



# ========== Bloom ============ #

prompt_bloom = ["""Dialogue entre un Alan et Marie. Marie a accès aux informations suivantes dans le réglement: 
        Début des informations : """,
        """
        - fin des informations à disposition de Marie.
        Le Alan pose la question : '""",
        " ?' Marie lui répond naturellement "]
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
    log_activity("bloom", user_question, output)
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
    log_activity("vicuna", user_question, answer)
    return answer