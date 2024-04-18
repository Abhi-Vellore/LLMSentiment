import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def load_model(model_path):
    # Initialize callback manager with appropriate handlers for monitoring or logging

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_model = LlamaCpp(
        model_path=model_path,
        temperature=1,
        max_tokens=10,
        top_p=0.1,
        n_gqa=8,
        callback_manager=callback_manager,
        verbose=True
    )
    return llama_model

def initialize_langchain(llama_model, template, input_variables):
    variable_names = list(input_variables.keys())
    prompt = PromptTemplate(input_variables=variable_names, template=template)
    llm_chain = LLMChain(prompt=prompt, llm=llama_model)
    return llm_chain

def validate_score(score):
    try:
        score = int(score.strip())  # Ensure the score is an integer and remove any extra whitespace
        if score < 1 or score > 5:
            raise ValueError("Score out of allowed range.")
        return score
    except ValueError:
        print(f"Invalid score received: {score}")
        return None

def analyze_sentiments(dataframe, llm_chain):
    results = []
    for index, row in dataframe.iterrows():
        try:
            result = llm_chain.invoke({"text": row['Text']})
            validated_score = validate_score(result)
            results.append(validated_score if validated_score is not None else 'Error')
        except Exception as e:
            print(f"An error occurred processing row {index}: {e}")
            results.append('Error')
    dataframe['llama-13-score'] = results
    return dataframe




