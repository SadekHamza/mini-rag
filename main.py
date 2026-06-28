from pypdf import PdfReader

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
for c in strToChunks(pdfToStr("user.pdf")):
    print(c)
    print("______________________")