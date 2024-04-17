import os
from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = "https://llmsentimentanalysis.openai.azure.com/"
    )
    
deployment_name='gpt35' 
    
# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. Make it exactly one sentence long '
response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=40)
print(start_phrase+response.choices[0].text)