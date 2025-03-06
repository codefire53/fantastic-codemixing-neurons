import random, os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import pickle

def save_pkl(acts, filename: str):
    with open(filename, 'wb') as handle:
        pickle.dump(acts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(filename: str):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

# get model and respective tokenizer
def initialize_model_and_tokenizer(model_name):
    access_token = "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"
    if 'aya' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    elif 'jais' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, adding_side='left', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    elif 't5' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    
    elif 'Qwen' in model_name or 'bloom' in model_name or 'OpenHathi' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

    elif 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            device_map="auto"
        )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
