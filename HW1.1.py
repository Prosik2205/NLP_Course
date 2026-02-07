#SETUP & INSTALLATION

import nltk
import spacy
import sklearn
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

print(f"NLTK version: {nltk.__version__}")
print(f"spaCy version: {spacy.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

# Test spaCy model
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello, world!")
print(f"spaCy model loaded: {nlp.meta['name']}")


#Ex1
#1.1

# Download required NLTK data (run once)
nltk.download('punkt')

# Sample text
text = """Natural Language Processing is fascinating. It enables computers
to understand human language.
Dr. Smith works at N.A.S.A. on text analysis. He said, "NLP is the
future!"
What do you think? Visit www.nlp.org for more info."""

# Tokenize into sentences
sentences = sent_tokenize(text)
print(f"Number of sentences: {len(sentences)}\n")
for i, sent in enumerate(sentences, 1):
 print(f"Sentence {i}: {sent}")