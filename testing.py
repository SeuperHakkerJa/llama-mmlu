import torch
import bitsandbytes as bnb

from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_name, bnb_config, ft=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if ft:
        model = prepare_model_for_kbit_training(mode)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # tokenizer.pad_token = tokenizer.eos_toekn
    # tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer


