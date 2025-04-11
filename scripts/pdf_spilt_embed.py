import re
import os
import pdfplumber
from langchain_ollama import OllamaEmbeddings  
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL

class pdfProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Extract filename
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)  
    

    def extract_text_from_pdf(self, pdf_path):

        full_text = []  

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)  


        return "\n".join(full_text)
    
    ## Split text into segments and return a list
    def split_sentences(self, text):
        """Support Chinese and English sentence segmentation (handling common abbreviations)"""
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(text) if s.strip()]
        return sentences
    
    def generate_id(self, index):
        """Generate ID according to requirements"""
        return f"{self.base_name}{index+1}"
    
    def process(self):
        text = self.extract_text_from_pdf(self.pdf_path)
        
        sentences = self.split_sentences(text)
        for idx, sent in enumerate(sentences):
            sent_id = self.generate_id(idx)
            self.sentence_to_id[sent] = sent_id
            self.id_to_sentence[sent_id] = sent
        
        vectors = self.embeddings.embed_documents(sentences)
        return {
            "sentences": sentences,
            "vectors": vectors,
            "sentence_to_id": self.sentence_to_id,
            "id_to_sentence": self.id_to_sentence

        }

# Usage example
processor = pdfProcessor("data/xxx.pdf")
result = processor.process()
