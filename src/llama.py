import pandas as pd
import openai
import time

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
        
        # Process the response
        sentiment_score = response.choices[0].message.content
        try:
            # Extract the first number in the response, which is assumed to be the sentiment score
            sentiment_score = int(sentiment_score.strip())
        except ValueError:
            # Handle cases where conversion to int fails
            print(f"Could not convert response to int: '{sentiment_score}'")
            sentiment_score = None
        
        return sentiment_score