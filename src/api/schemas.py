from pydantic import BaseModel
from typing import Optional, List, Any
from enum import Enum

class TrainingConfig(BaseModel):
    # LoRA attention dimension
    lora_r: Optional[int] = 64
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.1

    # bitsandbytes parameters
    use_4bit: Optional[bool] = True
    bnb_4bit_compute_dtype: Optional[str] = "float16"
    bnb_4bit_quant_type: Optional[str] = "nf4"
    use_nested_quant: Optional[bool] = False

    # TrainingArguments parameters
    output_dir: Optional[str] = "./artifacts/model"
    num_train_epochs: Optional[int] = 1
    fp16: Optional[bool] = False
    bf16: Optional[bool] = False
    per_device_train_batch_size: Optional[int] = 4
    per_device_eval_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 1
    gradient_checkpointing: Optional[bool] = True
    max_grad_norm: Optional[float] = 0.3
    learning_rate: Optional[float] = 2e-4
    weight_decay: Optional[float] = 0.001
    optim: Optional[str] = "paged_adamw_32bit"
    lr_scheduler_type: Optional[str] = "constant"
    max_steps: Optional[int] = -1
    warmup_ratio: Optional[float] = 0.03
    group_by_length: Optional[bool] = True
    save_steps: Optional[int] = 25
    logging_steps: Optional[int] = 25

    # SFT parameters
    max_seq_length: Optional[int] = None
    packing: Optional[bool] = False
    device_map: Optional[dict[str, int]] = {"": 0}


class TrainingRequest(BaseModel):
    model_name: str
    name_space: str
    dataset_name: str
    new_model: str
    hugging_face_api: str
    config: Optional[TrainingConfig] = None

class QuantizationConfig(BaseModel):
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

class ModelRequest(BaseModel):
    model_name: str
    name_space: str
    token: Optional[str] = None
    qunatization_config: Optional[QuantizationConfig] = None

class ChatRequest(BaseModel):
    query: str



