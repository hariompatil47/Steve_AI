from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def load_llm():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)