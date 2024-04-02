# src/preprocessing.py
import re
import pandas as pd

def clean_text(text):
   if pd.isnull(text):
      return ""
   text = re.sub('<.*?>', '', text)  # Remove HTML tags
   return text.lower()

def preprocess_reviews(df):
   df['reviewText'] = df['reviewText'].astype(str).apply(clean_text)
   df['reviewText'].fillna('', inplace=True)
   return df
