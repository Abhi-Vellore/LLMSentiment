import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import json
from config import DATA_PATH, PROCESSED_DATA_PATH



# Define a simple emoticon dictionary
emoticons = {
    ':)': ' smiley ',
    ':(': ' sad ',
}

class Preprocessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))

    def remove_stop_words(self, text):
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word.lower() not in self.stop_words])

    def handle_emoticons(self, text):
        for emoticon, replacement in emoticons.items():
            text = text.replace(emoticon, replacement)
        return text

    def preprocess_text(self, text):
        text = text.lower()
        text = self.remove_stop_words(text)
        text = self.handle_emoticons(text)
        return text

class DatasetPreprocessor(Preprocessor):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = f"{DATA_PATH}{file_path}"
        self.file_type = self._detect_file_type()


    def _detect_file_type(self):
        if self.file_path.endswith('.json'):
            return 'json'
        elif self.file_path.endswith('.csv'):
            return 'csv'
        elif self.file_path.endswith('.txt'):
            return 'txt'
        else:
            raise ValueError("Unsupported file type")

    def _read_json(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)

    def _read_csv(self):
        return pd.read_csv(self.file_path, encoding='ISO-8859-1')

    def _read_txt(self):
        data = []
        with open(self.file_path, 'r', encoding = "ISO-8859-1") as file:
            for line in file:
                if line.strip().endswith('@positive'):
                    sentence, label = line.rsplit('@positive', 1)
                    label = 'positive'
                elif line.strip().endswith('@negative'):
                    sentence, label = line.rsplit('@negative', 1)
                    label = 'negative'
                elif line.strip().endswith('@neutral'):
                    sentence, label = line.rsplit('@neutral', 1)
                    label = 'neutral'
                else:
                    print(f"Skipping line due to unexpected format: {line}")
                    continue
                data.append([sentence.strip(), label.strip()])
        return pd.DataFrame(data, columns=['sentence', 'sentiment'])

    def preprocess(self):
        if self.file_type == 'json':
            df = self._read_json()
        elif self.file_type == 'csv':
            df = self._read_csv()
        elif self.file_type == 'txt':
            df = self._read_txt()
        else:
            raise ValueError("Unsupported file type")
        
        self.df = self.preprocess_df(df)
        return self.df

        return self.preprocess_df(df)

    def preprocess_df(self, df):
        if hasattr(self, 'preprocess_data'):
            df = self.preprocess_data(df)
        else:
            raise NotImplementedError("preprocess_data method is not implemented in child class.")
        return df

    def to_csv(self, output_file_path):
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.makedirs(PROCESSED_DATA_PATH)

        # Join the processed directory path with the output file name
        output_file_path = os.path.join(PROCESSED_DATA_PATH, output_file_path)
        self.df.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

class AmazonDatasetPreprocessor(DatasetPreprocessor):
    def preprocess_data(self, df):
        df['Text'] = df['reviewText'].astype(str).apply(self.preprocess_text)
        return df

class KaggleDatasetPreprocessor(DatasetPreprocessor):
    def preprocess_data(self, df):
        df['Text'] = df['text'].astype(str).apply(self.preprocess_text)
        return df

class SentenceDatasetPreprocessor(DatasetPreprocessor):
    def preprocess_data(self, df):
        # Preprocess the sentences
        df['Text'] = df['sentence'].astype(str).self.preprocess_text)
        
        # Drop the original 'sentence' column as it's no longer needed
        df.drop('sentence', axis=1, inplace=True)

        df.rename(columns={'sentiment': 'score'}, inplace=True)

        return df
