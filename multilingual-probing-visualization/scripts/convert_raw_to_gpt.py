'''
Takes raw text and saves GPT-like models' features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu, Feb 2019, Ethan Chi, ethanchi@stanford.edu, May 2020
Modifications by Mahardika Krisna Ihsani, mahardika.ihsani@mbzuai.ac.ae

'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from argparse import ArgumentParser
import h5py
import numpy as np
from tqdm import tqdm
import pickle


argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('decoder_model')
argp.add_argument('language_code')
argp.add_argument('--neuron_file', type=str, default=None)
argp.add_argument('--component_to_patch', choices=['attention', 'mlp'], default=None)
args = argp.parse_args()

def merge_multiple_subwords(subword, is_beginning_subword):
    split_tokens = subword.split("Ġ")
    if is_beginning_subword:
        split_tokens = [f"Ġ{t}" if i > 0 else t  for i, t in enumerate(split_tokens) ]
    else:
        split_tokens = [f"Ġ{t}"  for i, t in enumerate(split_tokens) if i > 0]
    return split_tokens

class Deactivator:
    def __init__(self, target_mlp_layer, selected_neurons):
        self.target_mlp_layer = target_mlp_layer
        self.selected_neurons = selected_neurons
        self.deactivation_hook = target_mlp_layer.register_forward_hook(self.deactivate)
    
    def deactivate(self, module, input, output):
        output[:, :, self.selected_neurons] *= 0
        return output
    
    def remove_hook(self):
        self.deactivation_hook.remove()


def load_pkl(filename: str):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def initialize_model_tokenizer_and_config(model_name):
    access_token = "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"
    if 'aya' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        cfg = AutoConfig.from_pretrained(model_name, token=access_token)
    elif 'jais' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_name)
    elif 'Qwen' in model_name or 'bloom' in model_name or 'OpenHathi' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        cfg = AutoConfig.from_pretrained(model_name)
    elif 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            device_map="auto"
        )
        cfg = AutoConfig.from_pretrained(model_name, token=access_token)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, cfg

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
model, tokenizer, cfg = initialize_model_tokenizer_and_config(args.decoder_model)
LAYER_COUNT = cfg.num_hidden_layers
FEATURE_COUNT = cfg.hidden_size
patched_neurons_per_layer = None
if args.neuron_file is not None:
    patched_neurons_per_layer = load_pkl(args.neuron_file)
model.eval()
deactivators = []
if patched_neurons_per_layer is not None and args.component_to_patch == 'attention':
    with torch.no_grad():
        if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            for idx, layer in enumerate(model.model.layers):
                if idx not in patched_neurons_per_layer:
                    continue
                patched_neurons_list = patched_neurons_per_layer[idx]
                
                 # patch key neurons
                if 'key' in patched_neurons_list and len(patched_neurons_list['key'])  > 0:
                    proj_shape = layer.self_attn.k_proj.weight.shape[0]
                    query_shape = layer.self_attn.q_proj.weight.shape[0]
                    divider = query_shape//proj_shape
                    patched_neurons_list_key = list(set([neuron_pos//divider for neuron_pos in patched_neurons_list['key']]))
                    layer.self_attn.k_proj.weight[patched_neurons_list_key, :] = 0
                
                # patch query neurons
                if 'query' in patched_neurons_list and len(patched_neurons_list['query'])  > 0:
                    layer.self_attn.q_proj.weight[patched_neurons_list['query'], :] = 0
                
                # patch value neurons
                if 'value' in patched_neurons_list and len(patched_neurons_list['value'])  > 0:
                    proj_shape = layer.self_attn.v_proj.weight.shape[0]
                    query_shape = layer.self_attn.q_proj.weight.shape[0]
                    divider = query_shape//proj_shape
                    patched_neurons_list_value = list(set([neuron_pos//divider for neuron_pos in patched_neurons_list['value']]))
                    layer.self_attn.v_proj.weight[patched_neurons_list_value, :] = 0
                
                # patch output neurons
                if 'output' in patched_neurons_list and len(patched_neurons_list['output'])  > 0:
                    layer.self_attn.o_proj.weight[patched_neurons_list['output'], :] = 0
        
        else:
            for idx, layer in enumerate(model.transformer.h):
                if idx not in patched_neurons_per_layer:
                    continue
                patched_neurons_list = patched_neurons_per_layer[idx]

                multiplier_shape = layer.self_attention.query_key_value.weight.shape[0]//3

                 # patch key neurons
                if 'key' in patched_neurons_list and len(patched_neurons_list['key'])  > 0:
                    patched_neurons_list_key = [multiplier_shape+neuron_pos for neuron_pos in patched_neurons_list['key']]
                    layer.self_attention.query_key_value.weight[patched_neurons_list_key, :] = 0
                
                # patch query neurons
                if 'query' in patched_neurons_list and len(patched_neurons_list['query'])  > 0:
                    layer.self_attention.query_key_value.weight[patched_neurons_list['query'], :] = 0
                
                # patch value neurons
                if 'value' in patched_neurons_list and len(patched_neurons_list['value'])  > 0:
                    patched_neurons_list_value = [multiplier_shape*2+neuron_pos for neuron_pos in patched_neurons_list['value']]
                    layer.self_attention.query_key_value.weight[patched_neurons_list_value, :] = 0
                
                # patch output neurons
                if 'output' in patched_neurons_list and len(patched_neurons_list['output'])  > 0:
                    patched_neurons_list_output = [neuron_pos for neuron_pos in patched_neurons_list['output']]
                    layer.self_attention.dense.weight[patched_neurons_list_output, :] = 0



with h5py.File(args.output_path, 'a') as fout:
  for index, line in enumerate(tqdm(open(args.input_path))):
    key = args.language_code + '-' + str(index)
    line = line.strip() # Remove trailing characters

    tokenized_text = tokenizer.tokenize(line)
    new_tokenized_sent = []
    for idx, subword in enumerate(tokenized_text):
        if "Ġ" in subword:
            new_tokenized_sent.extend(merge_multiple_subwords(subword, idx==0))
        else:
            new_tokenized_sent.append(subword)
    tokenized_text = new_tokenized_sent
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]

    # Convert inputs to PyTorch tensorsf
    tokens_tensor = torch.tensor([indexed_tokens]).to(model.device)
    segments_tensors = torch.tensor([segment_ids])

    with torch.no_grad():


        encoded_layers = model(tokens_tensor, output_hidden_states=True).hidden_states
        encoded_layers = encoded_layers[1:]

        if patched_neurons_per_layer is not None:
            if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
                deactivators = [Deactivator(layer.mlp.act_fn, patched_neurons_per_layer[idx]) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
            elif 'BloomForCausalLM' in str(type(model)):
                deactivators = [Deactivator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx]) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
            else:
                print('model is not supported!')
                quit()

      
        for deactivator in deactivators:
            deactivator.remove_hook()
      

    key = args.language_code + '-' + str(index)
    try:
      dset = fout.create_dataset(key, (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    except RuntimeError:
      dset = fout[key]

    dset[:,:,:] = np.vstack([x.detach().cpu().float().numpy() for x in encoded_layers])

  print("Current keys are: ", ", ".join(fout.keys()))
