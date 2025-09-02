# Hugging Face Models using API :

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import util

def hugging_face_using_api():
    llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)

    result = model.invoke("What is the capital of India")

    print(result.content)
    
    
    
    """
We can use models that are present in Hugging Face
We just need to copy the model name and use it in model_id, first, it will download it and then use it
    """
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

def hugging_face_using_local_download():

    #To store downloaded models in specific directory
    os.environ['HF_HOME'] = 'D:\learning\llm_downloaded\hugging_face_cache'

    llm = HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=100
        )
    )
    model = ChatHuggingFace(llm=llm)

    result = model.invoke("What is the capital of India")

    print(result.content)

    
