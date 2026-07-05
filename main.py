from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pdfToStr(nomPdf):
    res=""
    pdf = PdfReader(nomPdf)
    for p in pdf.pages:
        res += p.extract_text()
    return res

def strToChunks(txt, chunkSize = 100):
    mots = txt.split()
    chunks = []
    for i in  range(0 , len(mots), chunkSize):
        chunks.append(" ".join(mots[i : i+chunkSize]))
    return chunks


class Search():
    def __init__(self, chunks ):
        self.documents = chunks
        self.vectorizer = TfidfVectorizer()
        self.tf_idf = self.vectorizer.fit_transform(chunks)
    def search(self ,query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec,self.tf_idf).flatten()
        indices = scores.argsort()[-top_k :][::-1]
        return [(self.documents[i], scores[i]) for i in indices]
    



