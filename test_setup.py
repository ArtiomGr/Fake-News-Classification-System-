import torch
import pandas as pd
import numpy as np
import sklearn
import transformers
import streamlit
import nltk

print("================================")
print("Fake News Project - Setup Test")
print("================================")

print("PyTorch:", torch.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Transformers:", transformers.__version__)
print("Streamlit:", streamlit.__version__)

print()
print("CUDA available:", torch.cuda.is_available())

print()
print("Everything is installed correctly!")
print("end of test_setup.py")