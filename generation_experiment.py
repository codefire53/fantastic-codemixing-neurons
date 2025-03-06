from hooks import Deactivator, Activator
from utils import initialize_model_and_tokenizer, seed_everything, load_neuron_acts, save_neuron_acts
import argparse
import torch
import os
from tqdm import tqdm


import torch.nn.functional as F
import einops
import numpy as np

PROMPT_TEMPLATE = "Please write down the output only in latin script. Generate only the translation. Do not output any extra test. Translate an English sentence into a Hindi-English codemixed sentence.\nEnglish: {text}\nHindi English: "

def generate_texts(inputs, tokenizer, model, args, seed=42):
    if args.experiment_type == 'unconditional':
        seed_everything(seed)
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            attention_mask = inputs.attention_mask.to(model.device),
            do_sample = True,
            temperature = args.temperature,
            top_p = args.top_p,
            max_length = args.max_output_length
        )
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_texts = [generated_texts[0]]

    else:
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            do_sample=False,
            max_length = args.max_output_length
        )
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def attach_mlp_hooks(model, patched_neurons_per_layer, hook_type, patch_neuron_start_indices, args):
    if hook_type == 'deact':
        if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            hooks = [Deactivator(layer.mlp.act_fn, patched_neurons_per_layer[idx], model.device, patch_neuron_start_indices=patch_neuron_start_indices) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
        elif 'BloomForCausalLM' in str(type(model)):
            hooks = [Deactivator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx], model.device, patch_neuron_start_indices=patch_neuron_start_indices) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
        else:
            print('model is not supported!')
            hooks = []
    else:
        hooks = []
        if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            hooks = [Activator(layer.mlp.act_fn, patched_neurons_per_layer[idx], patch_neuron_start_indices=patch_neuron_start_indices, multiplier_const=args.multiplier_const) for idx, layer in enumerate(model.model.layers) if idx in patched_neurons_per_layer]
        elif 'BloomForCausalLM' in str(type(model)):
            hooks = [Activator(layer.mlp.gelu_impl, patched_neurons_per_layer[idx], patch_neuron_start_indices=patch_neuron_start_indices, multiplier_const=args.multiplier_const) for idx, layer in enumerate(model.transformer.h) if idx in patched_neurons_per_layer]
        else:
            print('model is not supported!')
            hooks = []
    return hooks

def detach_hooks(hooks):
    for hook in hooks:
        hook.remove_hook()
    
def get_generated_texts(model, tokenizer, args, batch_size, patched_neurons_per_layer=None):
    out_deact = []
    out_act = []
    out_clean = []
    model.eval()
    with torch.no_grad():
        if args.experiment_type == 'unconditional':
            for sentence_idx in tqdm(range(args.num_of_output_samples)):
                prefix = tokenizer.bos_token
                inputs = tokenizer(prefix, return_tensors='pt')
                if patched_neurons_per_layer is not None:
                    # deactivators
                    if not args.ignore_deactivation:
                        deactivators = attach_mlp_hooks(model, patched_neurons_per_layer, 'deact')
                        deact_texts = generate_texts(inputs, tokenizer, model, args, sentence_idx)
                        detach_hooks(deactivators)
                        out_deact.extend(deact_texts)

                    # activator
                    activators = attach_mlp_hooks(model, patched_neurons_per_layer, 'act')
                    act_texts = generate_texts(inputs, tokenizer, model, args, sentence_idx)
                    detach_hooks(activators)
                    out_act.extend(act_texts)
                
                clean_texts = generate_texts(inputs, tokenizer, model, args, sentence_idx)
                out_clean.extend(clean_texts)
        else:
            texts = []
            with open(args.text_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    prompt = line.strip()
                    prompt = PROMPT_TEMPLATE.format(text=prompt)
                    prompt_order = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(prompt_order, tokenize=False, add_generation_prompt=True)
                    texts.append(prompt)
            
            for batch_idx in tqdm(range(0, len(texts), args.batch_size)):
                batch_texts = texts[batch_idx:min(batch_idx+args.batch_size, len(texts))]
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
                generated_text_indices = [len(input_ids)-1 for input_ids in inputs['input_ids']]
                if patched_neurons_per_layer is not None:
                    # deactivators
                    deactivators = attach_mlp_hooks(model, patched_neurons_per_layer, 'deact', generated_text_indices, args)
                    
                    deact_texts = generate_texts(inputs, tokenizer, model, args)
                    detach_hooks(deactivators)
                    out_deact.extend(deact_texts)

                    # activator
                    activators = attach_mlp_hooks(model, patched_neurons_per_layer, 'act', generated_text_indices, args)
                    act_texts = generate_texts(inputs, tokenizer, model, args)
                    detach_hooks(activators)
                    out_act.extend(act_texts)
                
                clean_texts = generate_texts(inputs, tokenizer, model, args)
                out_clean.extend(clean_texts)

    torch.cuda.empty_cache()
    return out_clean, out_act, out_deact

def load_data(filename: str):
    out_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            out_lines.append(line.lower().strip())
    return out_lines

def main(args):

    sample_path = f"{args.output_file_prefix}_clean.pkl"
    # if os.path.exists(sample_path):
    #     print("File exists")
    #     return

    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    patched_neurons_per_layer = None
    if args.neuron_file is not None:
        patched_neurons_per_layer = load_neuron_acts(args.neuron_file)
        texts_clean, texts_act, texts_deact = get_generated_texts(model, tokenizer, args, args.batch_size, patched_neurons_per_layer)
        clean_out = f"{args.output_file_prefix}_clean.pkl"
        act_out = f"{args.output_file_prefix}_activated_multiplier_{args.multiplier_const}.pkl"
        if not args.ignore_deactivation:
            deact_out = f"{args.output_file_prefix}_deactivated.pkl"
        #save_neuron_acts(texts_clean,clean_out)
        save_neuron_acts(texts_act, act_out)
        if not args.ignore_deactivation:
            save_neuron_acts(texts_deact, deact_out)
    else:
        texts_clean, _, _ = get_generated_texts(model, tokenizer, args, args.batch_size)
        clean_out = f"{args.output_file_prefix}_clean.pkl"
        save_neuron_acts(texts_clean, clean_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_file', type=str, default=None)
    parser.add_argument('--text_file', type=str, default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file_prefix', type=str)
    parser.add_argument('--experiment_type', type=str, choices=['conditional', 'unconditional'])
    parser.add_argument('--max_output_length', type=int, default=32)
    parser.add_argument('--num_outputs', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_of_output_samples', type=int, default=100)
    parser.add_argument('--multiplier_const', type=int, default=3)
    parser.add_argument('--ignore_deactivation', action='store_true')

    args = parser.parse_args()
    seed_everything(42)
    main(args)

