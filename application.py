import uvicorn
import os
import sys
import time
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from src.api.schemas import TrainingRequest
from src.api.schemas import TrainingConfig
from src.api.schemas import QuantizationConfig
from src.api.schemas import ChatRequest
from src.api.schemas import ModelRequest
from src.features.finetune.finetune import FineTune
from src.features.chat.inference import Model
from src.features.chat.inference import Inference
from fastapi import FastAPI


# Place holder text and api object
application = FastAPI()

@application.post("/lora_finetune")
async def finetune(request: TrainingRequest):
    try:
        request_data = request
        if not request_data.config:
            request_data.config = TrainingConfig()

        finetune_pipe = FineTune(finetune_req = request_data)
        finetune_pipe.train()
        return{"message": "Finetune Suscessfull"}

    except Exception as e:
        raise e

@application.post("/load_model")
async def switch_model(request: ModelRequest):
    global model, tokenizer
    try:
        request_data = request
        print("switch_request:", request_data)

        model_init = Model(request=request_data)
        model, tokenizer = model_init.model()
        return {"message": f"Model '{request_data.model_name}' loaded successfully"}
    except Exception as e:
        raise e

@application.post("/chat")
async def chat(request: ChatRequest):
    global model, tokenizer
    try:
        request_data = request

        print("chat:", request_data)

        if model is None or tokenizer is None:
            raise Exception("Model and tokenizer must be initialized first.")
        print("chat:", request_data)
        inference = Inference(model, tokenizer)
        start_time = time.time() 
        result = inference.predict(query=request_data.query)
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return {"generated_response": result, "inference_time":elapsed_time }
    except Exception as e:
        raise e 


# Run the FastAPI app
import uvicorn
from pyngrok import ngrok

# def run_app():
uvicorn.run("application:application", host="0.0.0.0", port=8000)
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# Start the FastAPI app in the background
# import threading
# threading.Thread(target=run_app).start()

# Expose the local server with ngrok
# public_url = ngrok.connect(8000)
# print(f"Public URL: {public_url}")
# if __name__ == "__main__":
#     import app
#     uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
