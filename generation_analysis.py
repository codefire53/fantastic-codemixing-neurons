from hooks import Deactivator, Activator
from utils import initialize_model_and_tokenizer, seed_everything, load_neuron_acts, save_neuron_acts
import argparse
import torch
import os
from tqdm import tqdm


import torch.nn.functional as F
import einops
import numpy as np
import json

PROMPT_TEMPLATE = "Please write down the output only in latin script. Generate only the translation. Do not output any extra test. Translate an English sentence into a Hindi-English codemixed sentence.\nEnglish: {text}\nHindi English: "

import numpy as np
from scipy.special import softmax, kl_div

def entropy_difference(logits1, logits2):
    p1 = softmax(logits1, axis=-1)  # Apply softmax along vocab axis
    p2 = softmax(logits2, axis=-1)
    res1 = -np.sum(p1 * np.log(p1 + 1e-10), axis=-1)  # Compute along vocab axis
    res2 = -np.sum(p2 * np.log(p1 + 1e-10), axis=-1) 
    res = res2-res1
    res = res.sum(axis=0)
    return res

def js_divergence(logits1, logits2):
    p1 = softmax(logits1, axis=-1)  # Apply softmax along vocab axis
    p2 = softmax(logits2, axis=-1)
    m = 0.5 * (p1 + p2)
    
    kl1 = np.sum(kl_div(p1, m), axis=-1)  # Compute KL-div along vocab axis
    kl2 = np.sum(kl_div(p2, m), axis=-1)
    
    res = 0.5 * (kl1 + kl2)  # JSD for each batch element
    res = res.sum(axis=0)
    return res

def attach_mlp_hooks(model, patched_neurons_per_layer, patch_neuron_start_indices, multiplier):

    hooks = []
    if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
        hooks = [Activator(layer.mlp.act_fn, patched_neurons_per_layer[idx], patch_neuron_start_indices=patch_neuron_start_indices, multiplier_const=multiplier) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
    elif 'BloomForCausalLM' in str(type(model)):
        hooks = [Activator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx], patch_neuron_start_indices=patch_neuron_start_indices, multiplier_const=multiplier) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
    else:
        print('model is not supported!')
        hooks = []
    return hooks

def detach_hooks(hooks):
    for hook in hooks:
        hook.remove_hook()
    
def get_prob_diff(model, tokenizer, args, batch_size, patched_neurons_per_layer, activation_multipliers):
    out_deact = []
    out_act = []
    out_clean = []
    model.eval()
    with torch.no_grad():
        texts = []
        with open(args.text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                prompt = line.strip()
                prompt = PROMPT_TEMPLATE.format(text=prompt)
                prompt_order = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(prompt_order, tokenize=False, add_generation_prompt=True)
                texts.append(prompt)
        scores_map = dict()
        for multiplier in activation_multipliers:
            scores_map[multiplier] = {
                'js_div':0.0,
                'entropy_diff':0.0
            }
        for batch_idx in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[batch_idx:min(batch_idx+args.batch_size, len(texts))]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
            generated_text_indices = [len(input_ids)-1 for input_ids in inputs['input_ids']]

            clean_logits = model(**inputs).logits[:,-1]
            clean_logits = clean_logits.detach().float().cpu().numpy()

            # activator
            for multiplier in activation_multipliers:
                activators = attach_mlp_hooks(model, patched_neurons_per_layer, generated_text_indices, multiplier)
                intervened_logits = model(**inputs).logits[:,-1]
                intervened_logits = intervened_logits.detach().float().cpu().numpy()
                ent_diff = entropy_difference(clean_logits, intervened_logits)
                js_div = js_divergence(clean_logits, intervened_logits)
                scores_map[multiplier]['js_div'] += js_div
                scores_map[multiplier]['entropy_diff'] += ent_diff
                detach_hooks(activators)
        for multiplier in activation_multipliers:
            scores_map[multiplier]['js_div'] /= len(texts)
            scores_map[multiplier]['entropy_diff'] /= len(texts)
    torch.cuda.empty_cache()
    return scores_map

def load_data(filename: str):
    out_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            out_lines.append(line.lower().strip())
    return out_lines

def main(args):
    multipliers = [2,3,4,5]
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    patched_neurons_per_layer = None
    all_prob_diffs_map = dict()
    for neuron_file in tqdm(args.neuron_files):
        patched_neurons_per_layer = load_neuron_acts(neuron_file)
        intervention_name = neuron_file.rsplit('/',1)[-1]
        intervention_name = intervention_name.replace('.pkl','')
        
        prob_diff_scores_map = get_prob_diff(model, tokenizer, args, args.batch_size, patched_neurons_per_layer, multipliers)
        for multiplier in prob_diff_scores_map.keys():
            key_name = f"{intervention_name}_{multiplier}"
            all_prob_diffs_map[key_name] = prob_diff_scores_map[multiplier]

    with open(args.output_file, 'w') as f:
        json.dump(all_prob_diffs_map, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_files', type=str, nargs='+')
    parser.add_argument('--text_file', type=str, default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str)



    args = parser.parse_args()
    seed_everything(42)
    main(args)

