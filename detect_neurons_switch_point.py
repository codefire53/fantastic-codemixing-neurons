from hooks import MLPSwitchActivationRecorder
from attention_hooks import BloomSelfAttentionWrapper, SelfAttentionWrapper
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import argparse
import einops
from utils import load_pkl, save_pkl, initialize_model_and_tokenizer, seed_everything
import gc
import numpy as np

def collect_acts(model, tokenizer, sentences_batch, switch_positions, nonswitch_positions, activation_method):
    model.eval()
    with torch.no_grad():
        if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            act_recorders = [MLPSwitchActivationRecorder(layer.mlp.act_fn, switch_positions, nonswitch_positions, activation_method) for layer in model.model.layers]
        elif 'BloomForCausalLM' in str(type(model)):
            act_recorders = [MLPSwitchActivationRecorder(layer.mlp.gelu_impl, switch_positions, nonswitch_positions, activation_method) for layer in model.transformer.h]
        elif 'JAISLMHeadModel' in str(type(model)):
            act_recorders = [MLPSwitchActivationRecorder(layer.mlp.act, switch_positions, nonswitch_positions, activation_method) for layer in model.transformer.h]
        else:
            act_recorders = []
            print('model is not supported!')
        tokenized_sentences = tokenizer(sentences_batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokenized_sentences.to(model.device)

        outputs = model(**tokenized_sentences)

        for act_recorder in act_recorders:
            act_recorder.remove_hook()
            
    switch_acts = torch.cat([act_recorder.layer_switch_point_acts.unsqueeze(0) for act_recorder in act_recorders], dim=0)
    nonswitch_acts = torch.cat([act_recorder.layer_nonswitch_point_acts.unsqueeze(0) for act_recorder in act_recorders], dim=0)

    return switch_acts, nonswitch_acts
    

def get_switching_and_non_switching_points(tokenizer, texts, switch_points, switch_indices):
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    switch_points_per_batch = []
    nonswitch_points_per_batch = []
    total_switch_points_cnt = 0
    total_nonswitch_points_cnt = 0
    for batch_pos in range(len(texts)):
        batch_switch_points = []
        batch_nonswitch_points = []

        curr_switch_points = switch_points[batch_pos]
        curr_switch_indices = switch_indices[batch_pos]

        assert len(curr_switch_indices) == len(curr_switch_indices)

        curr_input_ids = tokenized_inputs['input_ids'][batch_pos]
        curr_attn_mask = tokenized_inputs['attention_mask'][batch_pos]

        unique_curr_switch_points = list(set(curr_switch_points))
        tokenized_curr_switch_points = [tokenizer(substr)['input_ids'] for substr in unique_curr_switch_points]
        switch_points_len = [len(substr) for substr in tokenized_curr_switch_points]


        cand_per_switch_points = {}
        for substr in unique_curr_switch_points:
            cand_per_switch_points[substr] = []
        
        for start_idx in range(len(curr_input_ids)):
            if curr_attn_mask[start_idx] != 0:
                batch_nonswitch_points.append(start_idx)
        

        for start_idx in range(len(curr_input_ids)-1):
            if curr_attn_mask[start_idx] == 0:
                continue
            for sw_points, sw_points_len, sw_substr in zip(tokenized_curr_switch_points, switch_points_len, unique_curr_switch_points):
                max_spans_len = len(curr_input_ids)-start_idx
                if max_spans_len < sw_points_len:
                    continue
                
                if sw_points == curr_input_ids[start_idx:start_idx+sw_points_len]:
                    cand_per_switch_points[sw_substr].append(start_idx+sw_points_len-1) # we take the last token of a switch point word
        for sw_point, sw_index in zip(curr_switch_points, curr_switch_indices):
            if sw_index <= len(cand_per_switch_points[sw_point])-1:
                batch_switch_points.append(cand_per_switch_points[sw_point][sw_index])
        
        batch_nonswitch_points = [pos for pos in batch_nonswitch_points if pos not in batch_switch_points]
        switch_points_per_batch.append(batch_switch_points)
        total_nonswitch_points_cnt += len(batch_nonswitch_points)
        total_switch_points_cnt += len(batch_nonswitch_points)

        nonswitch_points_per_batch.append(batch_nonswitch_points)
    assert total_nonswitch_points_cnt > 0
    assert total_switch_points_cnt > 0
    return switch_points_per_batch, nonswitch_points_per_batch, total_switch_points_cnt, total_nonswitch_points_cnt
    
    
        

def get_act_diffs_avg(texts, switch_points, switch_indices, model, tokenizer, batch_size, activation_method='last'):
    avg_act_diffs = None
    total_switch_points_cnt = 0
    total_nonswitch_points_cnt = 0
    all_switch_acts = None
    all_nonswitch_acts = None
    for start_idx in tqdm(range(0, len(texts), batch_size)):
        end_pos = min(len(texts), start_idx+batch_size)
        lines_batch = texts[start_idx:end_pos]
        points_batch = switch_points[start_idx:end_pos]
        indices_batch = switch_indices[start_idx:end_pos]
        switching_points_batch, nonswitching_points_batch, batch_switch_points_cnt, batch_nonswitch_points_cnt = get_switching_and_non_switching_points(tokenizer, lines_batch, points_batch, indices_batch)
        total_switch_points_cnt += batch_switch_points_cnt
        total_nonswitch_points_cnt += batch_nonswitch_points_cnt
        switch_acts, nonswitch_acts = collect_acts(model, tokenizer, lines_batch, switching_points_batch, nonswitching_points_batch, activation_method)
        if all_switch_acts is None:
            all_switch_acts = switch_acts
        else:
            all_switch_acts += switch_acts
        
        if all_nonswitch_acts is None:
            all_nonswitch_acts = nonswitch_acts
        else:
            all_nonswitch_acts += nonswitch_acts
    all_nonswitch_acts /= total_nonswitch_points_cnt
    all_switch_acts /= total_switch_points_cnt
    return torch.abs(all_switch_acts-all_nonswitch_acts)


def get_sensitive_neurons(texts, switch_points, switch_indices, model, tokenizer, batch_size, activation_method):
    avg_act_diffs = get_act_diffs_avg(texts,switch_points, switch_indices, model, tokenizer, batch_size, activation_method)
    neurons_per_layer_cnt = avg_act_diffs.shape[-1]
    flattened_neuron_acts = einops.rearrange(avg_act_diffs, 'l n_neurons -> (l n_neurons)')
    sorted_neurons = torch.argsort(flattened_neuron_acts, descending=True)
    sorted_neurons_info = []
    for neuron in sorted_neurons:
        layer_pos = neuron//neurons_per_layer_cnt
        neuron_pos = neuron%neurons_per_layer_cnt
        neuron_name = f"L{layer_pos}N{neuron_pos}"
        sorted_neurons_info.append({
                'name': neuron_name,
                'layer': layer_pos.item(),
                'neuron_pos': neuron_pos.item(),
                'act_diff': flattened_neuron_acts[neuron].item()
        })
    return sorted_neurons_info


def load_text_data_feats(filepath: str):
    all_texts = []
    all_switch_points = []
    all_switch_indices = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        sep = '|'
        value_sep = ','
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            line = line.strip()
            features = line.split(sep)
            text = features[0]
            
            switch_points = features[1].split(value_sep)

            switch_points = [ele.replace("'",'').replace('"','') for ele in switch_points]
            new_switch_points = []
            for ele in switch_points:
                if ele == '<comma>':
                    new_switch_points.append(',')
                else:
                    new_switch_points.append(ele)
            switch_points = new_switch_points
            switch_indices = features[-1].split(value_sep)
            switch_indices = [int(ele) for ele in switch_indices]
            all_texts.append(text)
            all_switch_points.append(switch_points)
            all_switch_indices.append(switch_indices)
    return all_texts, all_switch_points, all_switch_indices

def main(args):
    acts = dict()
    if args.load_from_neuron_file:
        acts = load_pkl(args.neuron_file)
    


    batch_size = args.batch_size
    for model_name, dataset_file in tzip(args.model_names, args.dataset_files):
        texts, all_switch_points, all_switch_indices = load_text_data_feats(dataset_file)
        model_name_suffix = model_name.rsplit('/', 1)[-1]
        if model_name_suffix in acts:
            continue
        model, tokenizer = initialize_model_and_tokenizer(model_name)

        model_acts = get_sensitive_neurons(texts, all_switch_points, all_switch_indices, model, tokenizer, batch_size, args.activation_method)
        
        acts[model_name_suffix] = model_acts
        save_pkl(acts, args.neuron_file)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_file', type=str)
    parser.add_argument('--load_from_neuron_file', action='store_true')
    parser.add_argument('--dataset_files', type=str)
    parser.add_argument('--model_names', type=str)
    parser.add_argument('--activation_method', type=str, default='contrast')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    args.dataset_files = args.dataset_files.split(',')
    args.model_names = args.model_names.split(',')
    seed_everything(42)
    main(args)
