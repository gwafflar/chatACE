# LLM-pdf-chat
Repository for the course INFO-H512 current trends of AI. Purpose : use Large Language model to answer questions on PDF containing legal informations


#### Setup 
Get an OpenAI, an HuggingFace and a Replicate API keys and write it in a .env file (if missing , the associate Language Model will not be usable.)
```
OPENAI_API_KEY='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
API_KEY_HuggingFace='hf_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY'
REPLICATE_API_TOKEN = r8_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
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