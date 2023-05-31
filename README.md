# LLM-pdf-chat
Repository for the course INFO-H512 current trends of AI. Purpose : use Large Language model to answer questions on PDF containing legal informations


#### Setup 
Get an OpenAI API key and write it in a .env file
```
$echo "OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" > .env 
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