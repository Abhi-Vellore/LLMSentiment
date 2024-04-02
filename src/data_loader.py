import pandas as pd
import json
from config import DATA_PATH

# Directly define DATA_PATH here for testing

def load_json_data(file_name):
   data = []
   file_path = f"{DATA_PATH}{file_name}"
   with open(file_path, 'r', encoding='utf-8') as file:
      for line in file:
         data.append(json.loads(line))
   return pd.DataFrame(data)
