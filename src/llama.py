import pandas as pd
import openai
import time
import re

class LLaMaSession:
    def __init__(self, api_key, model, rate_limit_per_minute):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.llama-api.com")
        self.rate_limit_per_minute = rate_limit_per_minute
        self.last_request_time = None
    
    def analyze_sentiment(self, text):
        # Respect the rate limit
        if self.last_request_time is not None:
            time_since_last_request = time.time() - self.last_request_time
            time_to_wait = (60 / self.rate_limit_per_minute) - time_since_last_request
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        
        self.last_request_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "As a sentiment analysis model, rate the sentiment of the following text from 1 to 5, where 1 is very negative and 5 is very positive. Provide only the number as a response."},
                {"role": "user", "content": text}
            ]
        )
        
        # Check for 'choices' in the response and if it's not empty
        if 'choices' not in response or not response['choices']:
            print("No 'choices' found in response or 'choices' is empty.")
            return None

        # Check if 'message' and 'content' keys are in the response
        if 'message' not in response['choices'][0] or 'content' not in response['choices'][0]['message']:
            print("No 'message' or 'content' in 'choices' found in response.")
            return None

        # Use regular expression to find the first number in the response text
        match = re.search(r'\d+', response)
        if match:
            try:
                # Convert the found number to an integer
                sentiment_score = int(match.group())
            except ValueError:
                print(f"Could not convert found number to int: '{match.group()}'")
                sentiment_score = None
        else:
            print("No number found in the response text.")
            sentiment_score = None

        
        return sentiment_score
