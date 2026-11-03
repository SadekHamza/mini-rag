from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pdf_to_text(pdf_file):
    res=""
    pdf = PdfReader(pdf_file)
    for p in pdf.pages:
        p_text = p.extract_text()
        if (p_text == None):
            p_text = ""
        res += p_text + "\n"
    return res

def text_to_chunks(text, chunk_size=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")
    mots = text.split()
    chunks = []
    for i in  range(0 , len(mots), chunk_size):
        chunks.append(" ".join(mots[i : i+chunk_size]))
    return chunks


class TfidfSearch():
    def __init__(self, chunks ):
        if len(chunks) == 0 :
            raise ValueError("Impossible de construire le moteur de recherche : chunks est vide")
        
        self.documents = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

        if len(self.vectorizer.vocabulary_) == 0 :
            raise ValueError("Impossible de construire le moteur de recherche : vocabulary_ est vide")
    
    def search(self ,query, top_k=3):
    
        if top_k <= 0:
                raise ValueError("top_k doit être strictement positif.")
        top_k = min(top_k, len(self.documents))

        query_vector = self.vectorizer.transform([query]) # le vecteur de query peut etre nul donc on peut juger la question comme hors contexte
        if np.all(query_vector.toarray() == 0):
            return [("le vecteur de la question est nul",0)]
        similarity_scores = cosine_similarity(query_vector,self.tfidf_matrix).flatten()
        ranked_indices = similarity_scores.argsort()[-top_k :][::-1]
        return [(self.documents[i], similarity_scores[i]) for i in ranked_indices]


    



