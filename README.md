# ChatACE
Repository for the course INFO-H512 current trends of AI. 

Purpose : use Large Language model to answer questions from the Statutes and Internal Rules of the Association des Cercles Etudiants.


#### Setup 
Get an OpenAI, an HuggingFace and a Replicate API keys and write it in a .env file (if missing , the associate Language Model will not be usable.)
```
export OPENAI_API_KEY='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
export API_KEY_HuggingFace='hf_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY'
export REPLICATE_API_TOKEN = r8_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
```

Install the requirements : 
```
pip install -r requirements.txt
```

#### Run
After install `streamlit`, run the app using 

```
streamlit run chatACE.py
```