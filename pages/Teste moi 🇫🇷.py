import streamlit as st

from extract_and_clean_text import *
from language_and_ai import *

st.header("Pose une questions sur les Statuts ou ROI de l'ACE 💬""")
st.write("Pose une question à ton assistant virtuel. Tu as des questions sur les fonctionnements de l'ACE ? Tu veux savoir si tu peux ou non faire quelques choses ? Tu n'a pas envie de lire 50 pages de ROI ni les 30 pages de statuts pour le savoir ? Alors pose ta question ci-dessous ⬇️")
question_utilisateur = st.text_input("Pose une question:", placeholder="Quel est le rôle de l'Association des Cercles Etudiants ?") #change label and add transparent proposition ? 
if 'knowledge_base' not in st.session_state : 
        st.session_state['knowledge_base'] = "None"
if 'choice' not in st.session_state : #deprecated ? 
        st.session_state['choice'] = "None"

if question_utilisateur :
	if st.session_state['knowledge_base'] == "None" :
		analyze_text_ACE()
	knowledge_base = st.session_state['knowledge_base']
	st.session_state['choice'] = "ACE"
	docs = knowledge_base.similarity_search(question_utilisateur)
	try :
		response = generate_answer_from_OpenAI(docs, question_utilisateur)
		print(response)
		markdown_str = "### Réponse : " + response
		st.markdown(markdown_str)
	except : 
		st.error("Une erreur est apparue. Recommencer ou bien contactez l'administrateur de la page.")