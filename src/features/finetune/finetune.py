import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from src.api.schemas import TrainingRequest
from src.logging import logger

class FineTune:
    def __init__(self,
                 finetune_req:TrainingRequest) -> None:
        logger.info(f">>>>>> FineTuner Initializing with {finetune_req} <<<<<<")
        print("FineTuneReq: ", finetune_req)
        self.model_name = finetune_req.model_name
        self.name_space = finetune_req.name_space
        self.dataset_name = finetune_req.dataset_name
        self.new_model = finetune_req.new_model
        self.config = finetune_req.config
        self.hf_token = finetune_req.hugging_face_api
        print("model_name: ",self.model_name)
        print("name_space: ",self.name_space)
        print("dataset_name: ",self.dataset_name)
        print("new_model: ",self.new_model)
        print("config: ",self.config)
        print("hf_token: ",self.hf_token)
        
    def load_bits_and_bytes_config(self):
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
        return bnb_config,compute_dtype

    def set_training_arguments(self):
        training_arguments = TrainingArguments(
            output_dir="./artifacts/model_trainer/results",
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim=self.config.optim,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=self.config.group_by_length,
            lr_scheduler_type=self.config.lr_scheduler_type,
            report_to="tensorboard"
        )
        return training_arguments
    
    def load_lora_config(self):
        peft_config = LoraConfig(
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            r=self.config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return peft_config
    
    def push_to_hub(self,model,tokenizer,auth_token:str,new_model_name:str,namespace:str):

        try:
            import locale
            locale.getpreferredencoding = lambda: "UTF-8"
            directory = namespace + "/" + new_model_name
            model.push_to_hub(directory, check_pr=True, token=auth_token)
            tokenizer.push_to_hub(directory,check_pr=True, token=auth_token)
        except Exception as e:
            print(f"An error occurred while pushing to the repository: {str(e)}")

    def train(self):
        # Load dataset (you can process it here)
        logger.info(f">>>>>> Stage Loading DataSet started <<<<<<")
        dataset = load_dataset(self.dataset_name, split="train")
        logger.info(f">>>>>> Stage Loading DataSet ended <<<<<<")

        # Load tokenizer and model with QLoRA configuration

        logger.info(f">>>>>> Stage Load tokenizer and model with QLoRA configuration started <<<<<<")
        bnb_config,compute_dtype = self.load_bits_and_bytes_config()
        logger.info(f">>>>>> Stage Load tokenizer and model with QLoRA configuration bnb_config: {bnb_config}, compute_dtype: {compute_dtype}   ended <<<<<<")

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.config.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # Load base model
        logger.info(f">>>>>> Load Base Model Started <<<<<<")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        logger.info(f">>>>>> Load Base Model Ended <<<<<<")

        # Load LoRA configuration
        logger.info(f">>>>>> Load LoRA configuration Started <<<<<<")
        peft_config = self.load_lora_config()
        logger.info(f">>>>>> Load LoRA configuration: {peft_config} ended <<<<<<")

        # Set training parameters
        logger.info(f">>>>>> Set training parameters Started <<<<<<")
        training_arguments = self.set_training_arguments()
        logger.info(f">>>>>> Set training parameters Started <<<<<<")

        # Set supervised fine-tuning parameters
        logger.info(f">>>>>> Initialize SFTTrainer Started <<<<<<")
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=self.config.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.config.packing,
        )
        logger.info(f">>>>>> Initialize SFTTrainer: {trainer} ended <<<<<<")
        
        # Train model
        logger.info(f">>>>>> Train Model Started <<<<<<")
        trainer.train()
        logger.info(f">>>>>> Train Model Ended <<<<<<")

        # Save trained model
        logger.info(f">>>>>> Saving Trained Model Started <<<<<<")
        trainer.model.save_pretrained(os.path.join("artifacts/model_trainer/finetuned_model",self.new_model))
        # Empty VRAM
        del model
        del pipe
        del trainer
        import gc
        gc.collect()
        gc.collect()
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.config.device_map,
        )
        model = PeftModel.from_pretrained(base_model, os.path.join("artifacts/model_trainer/finetuned_model",self.new_model))
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info(f">>>>>> Saving Trained Model Ended <<<<<<")
        
        logger.info(f">>>>>> Pushing Model to HuggingFace Started<<<<<<")
        self.push_to_hub(model,tokenizer,auth_token=self.hf_token,new_model_name=self.new_model,namespace=self.name_space)
        logger.info(f">>>>>> Pushing Model to HuggingFace  Ended <<<<<<")
        

        
