# src/llama.py

import openai
import time
import re

class LLaMaSession:
    """Handles sessions with the LlaMA API including setting system content and sending prompts."""

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

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "As a sentiment analysis model, rate the sentiment of the following text from 1 to 5, where 1 is very negative and 5 is very positive. Provide only the number as a response."},
                    {"role": "user", "content": text}
                ]
            )
            # Access the relevant part of the response
            response_text = completion.choices[0].message.content
            match = re.search(r'\d+', response_text)
            if match:
                return int(match.group())
            else:
                print(f"No numeric response found in API response: {response_text}")
                return None
        except Exception as e:
            print(f"Error processing API response: {e}")
            return None
