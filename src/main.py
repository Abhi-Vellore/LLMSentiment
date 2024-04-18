import pandas as pd
from llama2 import load_model, initialize_langchain, analyze_sentiments

def main():
    # Define the model path and data path
    model_path = "llama-2-7b-chat.Q4_K_M.gguf"
    data_path = "test_experiment.csv"
    
    # Load the model
    llama_model = load_model(model_path)
    
    # Initialize the langchain with the sentiment analysis template
    template = "As a sentiment analysis model, rate the sentiment of the following text from 1 to 5, where 1 is very negative and 5 is very positive. Provide only the number as a response."
    input_variables = {'text': 'string'}  # Define the input variable type
    llm_chain = initialize_langchain(llama_model, template, input_variables)
    
    # Load the data
    data = pd.read_csv(data_path)
    
    # Perform sentiment analysis
    analyzed_data = analyze_sentiments(data, llm_chain)
    
    # Output results, for example, save to a new CSV
    output_path = "path_to_your_output.csv"
    analyzed_data.to_csv(output_path, index=False)
    print(f"Sentiment analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()



