from pre_process import AmazonDatasetPreprocessor, KaggleDatasetPreprocessor, SentenceDatasetPreprocessor
from config import PROCESSED_DATA_PATH
import os
import pandas as pd


def main():
   # Initialize and process the Amazon data
   amazon_processor = AmazonDatasetPreprocessor('Amazon_Fashion_Review_Data.json')
   amazon_processor.preprocess()
   amazon_processor.to_csv('processed_amazon_data.csv')

   # Initialize and process the Sentence data
   sentence_processor = SentenceDatasetPreprocessor('Sentences_75Agree.txt')
   sentence_processor.preprocess()
   sentence_processor.to_csv('processed_sentence_data.csv')

   # Initialize and process the first Kaggle dataset
   kaggle1_processor = KaggleDatasetPreprocessor('kaggle_train.csv')
   kaggle1_processor.preprocess()
   kaggle1_processor.to_csv('processed_kaggle_train_data.csv')

   # Initialize and process the second Kaggle dataset
   kaggle2_processor = KaggleDatasetPreprocessor('kaggle_test.csv')
   kaggle2_processor.preprocess()
   kaggle2_processor.to_csv('processed_kaggle_test_data.csv')

   # Concatenate the preprocessed DataFrames
   processed_kaggle1_df = kaggle1_processor.df
   processed_kaggle2_df = kaggle2_processor.df
   combined_df = pd.concat([processed_kaggle1_df, processed_kaggle2_df], ignore_index=True)

   # Save the combined DataFrame to a new CSV file in the processed directory
   # Since combined_df is not associated with a processor, we need to handle the path manually
   combined_csv_path = os.path.join(PROCESSED_DATA_PATH, 'processed_kaggle_combined_data.csv')
   combined_df.to_csv(combined_csv_path, index=False)
   print(f"Processed combined Kaggle data saved to {combined_csv_path}")

if __name__ == "__main__":
   main()
