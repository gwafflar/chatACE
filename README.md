# LLM-pdf-chat
Repository for the course INFO-H512 current trends of AI. Purpose : use Large Language model to answer questions on PDF containing legal informations


#### Setup 
Get an OpenAI and HuggingFace API keys and write it in a .env file
```
OPENAI_API_KEY='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
API_KEY_HuggingFace='hf_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY'
```

Install the requirements : 
```
pip install -r requirements.txt
```

#### Run
After install `streamlit`, run the app using 

```
streamlit run llm_ai_projet.py
```