import streamlit as st
from main import pdf_to_text, text_to_chunks, TfidfSearch

st.title("Premiere version")

uploaded_file = st.file_uploader( "Choisis un fichier PDF",type=["pdf"])
question = st.text_input("Quelle est ta question ?")

if st.button("Rechercher"):

    if uploaded_file is None:
        st.warning("Tu dois d'abord charger un PDF")

    elif question == "":
        st.warning("Tu dois ecrire une question")

    else:
        text = pdf_to_text(uploaded_file)
        chunks = text_to_chunks(text)
        if len(chunks) == 0:
            st.warning("aucun texte exploitable dans ce PDF")

        else:
            tfidf_search = TfidfSearch(chunks)
            results = tfidf_search.search(question, 1)
            document , score = results[0]
            st.subheader("Resultat le plus pertinent :")
            st.write(document)
            st.write("Score de similarité : "+ str(score))
