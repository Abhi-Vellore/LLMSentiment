# sentiment_analysis.py
import pandas as pd
import requests

class SentimentAnalysisSession:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {self.api_key}'}
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.system_message = {
            "role": "system",
            "content": "As a sentiment analysis AI, you will assign a score from 1 to 5 to the texts I provide, where 1 is absolutely negative and 5 is absolutely positive."
        }
        self.messages = [self.system_message]
        self.total_tokens_used = 0

    def add_user_message(self, text):
        self.messages.append({
            "role": "user",
            "content": text
        })

    def get_sentiment_score(self):
        data = {'model': 'gpt-3.5-turbo', 'messages': self.messages}
        response = requests.post(self.endpoint, headers=self.headers, json=data)
        
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}: {response.text}")
            return None
        
        response_data = response.json()
        choice = response_data['choices'][0]
        
        # Calculate token usage for this response
        tokens_used = len(choice['message']['content'].split())
        self.total_tokens_used += tokens_used
        print(f"Tokens used for this request: {tokens_used}")
        print(f"Total tokens used so far: {self.total_tokens_used}")

        # Reset the messages to just the system message after each call
        self.messages = [self.system_message]
        return choice['message']['content']

