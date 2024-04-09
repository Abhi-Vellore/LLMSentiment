# sentiment_analysis_session.py
import requests

class SentimentAnalysisSession:
    def __init__(self, openai_api_key):
        self.api_key = openai_api_key
        self.session_prompt = "Rate the sentiment of the following texts from 1 to 5, where 1 is absolutely negative and 5 is absolutely positive."

    def analyze_sentiment(self, text):
        combined_prompt = f"{self.session_prompt}\n\n{text}"
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': 'gpt-4',  # Adjust based on your model choice
            'messages': [{"role": "system", "content": self.session_prompt},
                        {"role": "user", "content": text}],
            'temperature': 1,
            'max_tokens': 10,
        }
        response = requests.post('https://api.openai.com/v1/chat/completions',
                                headers=headers, json=data)
        try:
            response_data = response.json()
            # Extracting the sentiment score from the response
            score = response_data['choices'][0]['message']['content']
            return score.strip()
        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return "Error"

