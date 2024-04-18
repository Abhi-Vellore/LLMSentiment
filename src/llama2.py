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
        temperature=0.7,
        max_tokens=512,
        top_p=1.0,
        callback_manager=callback_manager,
        verbose=True
    )
    return llama_model

def initialize_langchain(llama_model, template, input_variables):
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    llm_chain = LLMChain(prompt=prompt, llm=llama_model)
    return llm_chain

def analyze_sentiments(dataframe, llm_chain):
    dataframe['Results'] = dataframe.apply(
        lambda x: llm_chain.run({"text": x['text']}),
        axis=1
    )
    return dataframe

