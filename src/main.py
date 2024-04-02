# src/main.py
from data_loader import load_json_data
from preprocessing import preprocess_reviews
from config import PROCESSED_DATA_PATH

def main():
   # Load data
   df = load_json_data("Amazon_Fashion_Review_Data.json")

   # Preprocess data
   df_processed = preprocess_reviews(df)

   # Save processed data
   processed_file_path = f"{PROCESSED_DATA_PATH}cleaned_amazon_fashion_reviews.csv"
   df_processed.to_csv(processed_file_path, index=False)
   print(f"Processed data saved to {processed_file_path}")

if __name__ == "__main__":
   main()
