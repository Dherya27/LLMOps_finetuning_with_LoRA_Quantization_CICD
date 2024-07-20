import torch
# from transformers import whoami
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import pipeline
from src.logging import logger
from src.api.schemas import ModelRequest
from src.api.schemas import QuantizationConfig
from src.api.schemas import ChatRequest

class Model:
    def __init__(self, request: ModelRequest) -> None:
        self.qconfig = request.qunatization_config
        self.namespace = request.name_space
        self.model_name = request.model_name
        self.token = request.token

    def authenticate(token: str):
        # user = whoami(token=token)
        pass

    def load_quantization_config(self, config: QuantizationConfig):
        config = config
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(    
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        return bnb_config

    def load_hf_model(self, name_space:str, model_name:str, auth_key:str=None):
        try:
            model_path = name_space + "/" + model_name
            if auth_key:
                model = AutoModel.from_pretrained(model_path,token = auth_key)
                tokenizer = AutoTokenizer.from_pretrained(model_path,token = auth_key)
                return model, tokenizer
            else:
                model = AutoModel.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
        except Exception as e:
            logger.info(f"Exception {e} occured while loading model")

    def load_hf_model_quantization(self, name_space:str, model_name:str, quant_config: QuantizationConfig, auth_key:str=None):
        try:
            model_path = name_space + "/" + model_name
            if auth_key:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config =self.load_quantization_config(config=quant_config),
                    device_map = {"": 0},
                    token = auth_key
                    )
                model.config.pretraining_tp = 1
                tokenizer = AutoTokenizer.from_pretrained(model_path,token = auth_key)
                return model, tokenizer
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config =self.load_quantization_config(config=quant_config),
                    device_map = {"": 0}
                    )
                model.config.pretraining_tp = 1
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
        except Exception as e:
            logger.info(f"Exception {e} occured while loading model")

    def model(self):
        if self.token:
            # user = whoami(token=self.token)
            pass

        if self.qconfig:
            model, tokenizer = self.load_hf_model_quantization(
                name_space=self.namespace,
                model_name=self.model_name,
                quant_config=self.qconfig
            )
        else:
            model, tokenizer = self.load_hf_model(
                name_space=self.namespace,
                model_name=self.model_name
            )
        return model, tokenizer 
    
class Inference:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        import locale
        locale.getpreferredencoding = lambda: "UTF-8"
        
    def predict(self, query:ChatRequest):
        try:
            prompt = query
            pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=200)
            result = pipe(f"<s>[INST] {prompt} [/INST]")
            logger.info(f"Exception {result} occured while loading model")
            return result
        except Exception as e:
            logger.info(f"Exception {e} occured while loading model")
    


        