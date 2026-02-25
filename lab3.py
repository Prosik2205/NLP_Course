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


from sklearn.datasets import get_data_home
print(get_data_home())