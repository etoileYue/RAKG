import re
import os
import pdfplumber
from langchain_ollama import OllamaEmbeddings  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL

class TextProcessor:
    def __init__(self, text, name):
        self.text = text
        self.base_name = name # Extract filename
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)  
    
    
    ## Split text into segments and return a list
    def split_sentences(self, text):
        """Support Chinese and English sentence segmentation (handling common abbreviations)"""
        # Added Chinese punctuation (references 4, 5)
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(text) if s.strip()]
        return sentences
    
    def generate_id(self, index):
        """Generate ID according to requirements"""
        return f"{self.base_name}{index+1}"  # Start numbering from 1
    
    def process(self):
        # Step 1: Convert PDF to text
        text = self.text
        
        # Step 2: Sentence segmentation and ID mapping
        sentences = self.split_sentences(text)
        for idx, sent in enumerate(sentences):
            sent_id = self.generate_id(idx)
            self.sentence_to_id[sent] = sent_id
            self.id_to_sentence[sent_id] = sent
        
        # Step 3: Vector storage
        vectors = self.embeddings.embed_documents(sentences)
        return {
            "sentences": sentences,
            "vectors": vectors,
            "sentence_to_id": self.sentence_to_id,
            "id_to_sentence": self.id_to_sentence

        }
    


