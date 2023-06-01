from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
import streamlit as st



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