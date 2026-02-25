# Part 1: Setup and Data Exploration
# Step 1.1: Import Libraries

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Scikit-learn imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
 accuracy_score,
 classification_report,
 confusion_matrix,
 ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')
print("Libraries imported successfully!")

# Step 1.2: Load Dataset
# Define categories (subset for speed)
categories = [
 'alt.atheism', # Religion/philosophy discussions
 'comp.graphics', # Computer graphics
 'sci.space', # Space science and astronomy
 'talk.religion.misc' # General religion discussions
]
# Load training data
train_data = fetch_20newsgroups(
 subset='train',
 categories=categories,
 shuffle=True,
 random_state=42,
 remove=('headers', 'footers', 'quotes') # Remove metadata to prevent leakage
)
# Load test data
test_data = fetch_20newsgroups(
 subset='test',
 categories=categories,
 shuffle=True,
 random_state=42,
 remove=('headers', 'footers', 'quotes')
)
print(f"Training samples: {len(train_data.data)}")
print(f"Test samples: {len(test_data.data)}")
print(f"Categories: {train_data.target_names}")

# Step 1.3: Explore the Data

# Display a sample document
print("\n=== Sample Document ===")
print(f"Category: {train_data.target_names[train_data.target[0]]}")
print(f"Text (first 500 chars):\n{train_data.data[0][:500]}...")
# Check class distribution
print("\n=== Class Distribution (Training) ===")
unique, counts = np.unique(train_data.target, return_counts=True)
for label, count in zip(unique, counts):
 print(f"{train_data.target_names[label]:<25} {count:>4} samples")
# Calculate document length statistics
doc_lengths = [len(doc.split()) for doc in train_data.data]
print(f"\n=== Document Length Statistics ===")
print(f"Min length: {min(doc_lengths):>5} words")
print(f"Max length: {max(doc_lengths):>5} words")
print(f"Mean length: {np.mean(doc_lengths):>5.1f} words")
print(f"Median length: {np.median(doc_lengths):>5.1f} words")

