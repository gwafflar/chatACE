# ChatACE
Repository for the course INFO-H512 current trends of AI. 

Purpose : use Large Language model to answer questions from the Statutes and Internal Rules of the Association des Cercles Etudiants.

## Use

This project is hosted on Streamlit Community Cloud at the address : [chat-ace.streamlit.app](https://chat-ace.streamlit.app/)

Alternatively, you can also run it locally by following the instructions. However, this solution requires some API keys.


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
You can then access to [http://localhost:8501/](http://localhost:8501/) from your navigator.


## Screenshots
Page `Teste-moi`
![Page test-moi](images/screen_teste_moi_alcool.png)

Page `Explications - Explanations`
![Page Explanation](images/screen_explanation.png)

Page `chatACE` (main page)
![Main Page](images/screen_main_page_1.png)
![Main Page](images/screen_main_page_2.png)