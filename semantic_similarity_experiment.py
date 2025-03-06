from hooks import Deactivator
from utils import initialize_model_and_tokenizer, seed_everything, load_pkl, save_pkl
import argparse
import torch
import os
import numpy as np


import torch.nn.functional as F
from tqdm import tqdm
import einops
import numpy as np

def collect_semantic_sim(model, tokenizer, inputs_1, inputs_2, batch_size, ablation_method, patched_neurons_per_layer=None, mean_ablation_map=None):
    all_cos_sims = []
    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(inputs_1), batch_size)):
            deactivators = []
            if patched_neurons_per_layer is not None:
                if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
                    if 'zero' not in ablation_method:
                        deactivators = [Deactivator(layer.mlp.act_fn, patched_neurons_per_layer[idx], model.device, ablation_method, mean_ablation_map[idx]) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
                    else:
                        deactivators = [Deactivator(layer.mlp.act_fn, patched_neurons_per_layer[idx], model.device, ablation_method) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
                elif 'BloomForCausalLM' in str(type(model)):
                    if 'zero' not in ablation_method:
                        deactivators = [Deactivator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx], model.device, ablation_method, mean_ablation_map[idx]) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
                    else:
                        deactivators = [Deactivator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx], model.device, ablation_method) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
                else:
                    print('model is not supported!')
                    return []
            text_batch_1 = inputs_1[batch_start:min(batch_start+batch_size,len(inputs_1))]
            tokenized_batch_1 = tokenizer(text_batch_1, padding=True, return_tensors='pt')
            attention_mask_1 = tokenized_batch_1['attention_mask']
            input_ids_1 = tokenized_batch_1['input_ids'].to(model.device)
            attention_mask_1 = attention_mask_1.to(model.device)
            last_hidden_state_1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True).hidden_states[-1].detach().cpu()
            last_token_representation_1  = last_hidden_state_1[:, -1, :].unsqueeze(1)

         
            text_batch_2 = inputs_2[batch_start:min(batch_start+batch_size,len(inputs_1))]
            tokenized_batch_2 = tokenizer(text_batch_2, padding=True, return_tensors='pt')
            attention_mask_2 = tokenized_batch_2['attention_mask']
            input_ids_2 = tokenized_batch_2['input_ids'].to(model.device)
            attention_mask_2 = attention_mask_2.to(model.device)
            last_hidden_state_2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True).hidden_states[-1].detach().cpu()

            for deactivator in deactivators:
                deactivator.remove_hook()

            last_token_representation_2  = last_hidden_state_2[:, -1, :].unsqueeze(1)
            dot_prod = einops.einsum(last_token_representation_2, last_token_representation_1, 'b t d, b t d -> b').detach().cpu()

            norm_1 = torch.norm(last_token_representation_1, dim=[-2,-1]).detach().cpu()
            norm_2 = torch.norm(last_token_representation_1, dim=[-2,-1]).detach().cpu()
            eps = 1e-8
            normalizer = norm_1*norm_2
            cos_sim = torch.divide(dot_prod, normalizer)
            cos_sim_lst = [float(val.item()) for val in cos_sim]
            all_cos_sims.extend(cos_sim_lst)
    torch.cuda.empty_cache()
    return all_cos_sims

def load_data(filename: str):
    out_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            out_lines.append(line.lower().strip())
    return out_lines

def main(args):
    if os.path.exists(args.output_file):
        print("File exists")
        return
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    model_name_suffix = args.model_name.rsplit('/', 1)[-1]
    patched_neurons_per_layer = None
    if args.neuron_file is not None:
        patched_neurons_per_layer = load_pkl(args.neuron_file)
    mean_ablation_map = None
    if args.mean_ablation_file is not None:
        mean_ablation_map = load_pkl(args.mean_ablation_file)
        mean_ablation_map  = mean_ablation_map[model_name_suffix]
    inputs_1 = load_data(args.input_file_1)
    inputs_2 = load_data(args.input_file_2)
    ablation_method = args.ablation_method
    all_sem_scores = collect_semantic_sim(model, tokenizer, inputs_1, inputs_2, args.batch_size, ablation_method, patched_neurons_per_layer, mean_ablation_map)
    save_pkl(all_sem_scores, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_file', type=str, default=None)
    parser.add_argument('--mean_ablation_file', type=str, default=None)
    parser.add_argument('--input_file_1', type=str)
    parser.add_argument('--input_file_2', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--ablation_method', type=str, default='zero')
    args = parser.parse_args()
    seed_everything(42)
    main(args)

