import streamlit as st
from main import pdf_to_text, text_to_chunks, TfidfSearch

st.title("Premiere version")

uploaded_file = st.file_uploader( "Choisis un fichier PDF",type=["pdf"])
question = st.text_input("Quelle est ta question ?")

if st.button("Rechercher"):

    if uploaded_file is None:
        st.warning("Tu dois d'abord charger un PDF.")

    elif question.strip() == "":
        st.warning("Tu dois ecrire une question.")

    else:
        text = pdf_to_text(uploaded_file)
        chunks = text_to_chunks(text)
            

        try :
            tfidf_search = TfidfSearch(chunks)

        # une ValueError est declancher si la liste des documents\chunks extraits est vide ou fit_transform ne trouve pas de vocabulaire est donc vocabulary_ est vide ex dans le cas de chunks = ["a b $", "c d"]
        except ValueError as e: 
            st.warning(e)
            
        else :    
            results = tfidf_search.search(question, 3)

            # tester si le milleur score est nul ou si le vecteur de la questione est nul 
            if results[0][1] == 0:
                st.info("Aucune information trouvée dans le document")

            # afficher les 3 milleures documents\chunks avec leurs scores respectives
            else :
                for document , score in results :
                    st.write(document)
                    st.write("Score de similarité : "+ str(score))
