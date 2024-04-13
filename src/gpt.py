import pandas as pd
import requests


class ChatGPTSession:
    def __init__(self, api_key, model, rate_limit_per_minute):
        self.api_key = api_key
        self.model = model
        self.headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        self.context_set = False
        self.rate_limit_per_minute = rate_limit_per_minute
        self.last_request_time = None

    def set_context(self, context):
        self.context = context
        self.context_set = True

    def send_prompt(self, prompt):
        if not self.context_set:
            raise ValueError("Context not set. Call set_context before sending prompts.")
        
        messages = [{"role": "system", "content": self.context}]
        messages.append({"role": "user", "content": prompt})
        
        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 10,
            'temperature': 1,
            'top_p': 0.1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
        }
        
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=self.headers, json=data)
        response_data = response.json()
        

        # Assuming the API response contains a number as a sentiment score
        score = response_data['choices'][0]['message']['content']
        
        try:
            # Extract the first number in the response, which is the sentiment score
            sentiment_score = int(score.split()[0])
            return sentiment_score
        except ValueError:
            # Handle cases where conversion to int fails
            print(f"Could not convert response to int: {score}")
            return None
