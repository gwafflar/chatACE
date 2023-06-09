import streamlit as st

from extract_and_clean_text import *
from language_and_ai import *

#st.set_page_config(page_title="Pose tes questions sur l'ACE")
#st.sidebar.markdown("""# Teste moi "\
#                    Version simple et directe en français"
st.header("Pose une questions sur les Statuts ou ROI de l'ACE 💬""")
st.write("Pose une question à ton assistant virtuel. Tu as des questions sur les fonctionnements de l'ACE ? Tu veux savoir si tu peux ou non faire quelques choses ? Tu n'a pas envie de lire 50 pages de ROI pour le savoir ? Alors pose ta question ci-dessous ⬇️")
question_utilisateur = st.text_input("Pose une question:", placeholder="Quel est le rôle de l'Association des Cercles Etudiants ?") #change label and add transparent proposition ? 

if question_utilisateur :
	if st.session_state['knowledge_base'] == "None" :
		analyze_text_ACE()
	knowledge_base = st.session_state['knowledge_base']
	docs = knowledge_base.similarity_search(question_utilisateur)
	response = generate_answer_from_OpenAI(docs, question_utilisateur)
	print(response)
	markdown_str = "#### Réponse : " + response
	st.markdown(markdown_str)
