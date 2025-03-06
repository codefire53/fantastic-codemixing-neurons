from hooks import MLPActivationRecorder
from attention_hooks import BloomSelfAttentionWrapper, SelfAttentionWrapper
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import argparse
import einops
from utils import load_pkl, save_pkl, initialize_model_and_tokenizer, seed_everything
import gc
import numpy as np

def collect_acts(model, tokenizer, sentences_batch, module_type, activation_method='last'):
    model.eval()
    with torch.no_grad():
        if module_type == 'mlp':
            if 'MT5ForConditionalGeneration' in str(type(model)):
                act_recorders = [MLPActivationRecorder(layer.layer[-1].DenseReluDense.act, activation_method) for layer in model.encoder.block]
            elif 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
                act_recorders = [MLPActivationRecorder(layer.mlp.act_fn, activation_method) for layer in model.model.layers]
            elif 'BloomForCausalLM' in str(type(model)):
                act_recorders = [MLPActivationRecorder(layer.mlp.gelu_impl, activation_method) for layer in model.transformer.h]
            elif 'JAISLMHeadModel' in str(type(model)):
                act_recorders = [MLPActivationRecorder(layer.mlp.act, activation_method) for layer in model.transformer.h]
            else:
                act_recorders = []
                print('model is not supported!')
            tokenized_sentences = tokenizer(sentences_batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            tokenized_sentences.to(model.device)

            outputs = model(**tokenized_sentences)

            for act_recorder in act_recorders:
                act_recorder.remove_hook()
                
            acts = torch.cat([act_recorder.layer_outputs[0].unsqueeze(0) for act_recorder in act_recorders], dim=0)
        
        # get attention importance
        else:
            if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
                attn_imp_wrappers = [SelfAttentionWrapper(layer.self_attn, idx, model.device) for idx, layer in enumerate(model.model.layers)]
                for idx, layer in enumerate(model.model.layers):
                    layer.self_attn = attn_imp_wrappers[idx]
            else:
                attn_imp_wrappers = [BloomSelfAttentionWrapper(layer.self_attention, idx, model.device) for idx, layer in enumerate(model.transformer.h)]
                for idx, layer in enumerate(model.transformer.h):
                    layer.self_attention = attn_imp_wrappers[idx]
            
            # for idx, layer in enumerate(model.model.layers):
            #     breakpoint()

            tokenized_sentences = tokenizer(sentences_batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            tokenized_sentences.to(model.device)
            outputs = model(**tokenized_sentences)
            if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
                for idx, layer in enumerate(model.model.layers):
                    layer.self_attn = attn_imp_wrappers[idx].original_self_attn
            else:
                for idx, layer in enumerate(model.transformer.h):
                    layer.self_attention = attn_imp_wrappers[idx].original_self_attn

            # for idx, layer in enumerate(model.model.layers):
            #     breakpoint() 
            q_imp = torch.cat([act_recorder.query_importance for act_recorder in attn_imp_wrappers], dim=0)
            k_imp = torch.cat([act_recorder.key_importance for act_recorder in attn_imp_wrappers], dim=0)

            o_imp = torch.cat([act_recorder.output_importance for act_recorder in attn_imp_wrappers], dim=0)
            v_imp = torch.cat([act_recorder.value_importance for act_recorder in attn_imp_wrappers], dim=0)

        
            acts = {
                'key': k_imp,
                'query': q_imp,
                'value': v_imp,
                'output': o_imp
            }


        #print(acts.shape)
    # breakpoint()
    return acts


def get_act_diffs_avg(data, contrasting_data, model, tokenizer, module_type, batch_size, activation_method='last'):
    avg_act_diffs = None
    if len(contrasting_data) > 0:
        for start_idx in tqdm(range(0, len(data), batch_size)):
            lines_batch = data[start_idx:min(len(data), start_idx+batch_size)]
            contrasting_lines_batch = contrasting_data[start_idx:min(len(data), start_idx+batch_size)]
            acts = collect_acts(model, tokenizer, lines_batch, module_type, activation_method) #batch*hidden_dim
            contrasting_act_mean = None
            num_of_types = len(contrasting_lines_batch[0])
            #breakpoint()
            for instance_idx in range(num_of_types):
                instances_batch = [contrasting_lines_batch[pos_idx][instance_idx] for pos_idx in range(len(contrasting_lines_batch))]
                contrasting_acts = collect_acts(model, tokenizer, instances_batch, module_type, activation_method)
                if module_type == 'mlp':
                    if contrasting_act_mean is None:
                        contrasting_act_mean = torch.zeros_like(contrasting_acts)
                    contrasting_act_mean = contrasting_act_mean + contrasting_acts
                else:
                    if contrasting_act_mean is None:
                        contrasting_act_mean = dict()
                        for key in acts.keys(): 
                            contrasting_act_mean[key] = torch.zeros_like(acts[key])
                    for key in acts.keys():
                        contrasting_act_mean[key] += contrasting_acts[key]
            if module_type == 'mlp':
                contrasting_act_mean = contrasting_act_mean/num_of_types
                acts_diff = torch.abs(acts-contrasting_act_mean).sum(dim=-2)

                if avg_act_diffs is None:
                    avg_act_diffs = acts_diff
                else:
                    avg_act_diffs += acts_diff

            else:
                acts_diff = dict()
                for key in acts.keys():
                    contrasting_act_mean[key] = contrasting_act_mean[key]/num_of_types
                    acts_diff[key] = torch.abs(acts[key]-contrasting_act_mean[key])

                if avg_act_diffs is None:
                    avg_act_diffs = acts_diff
                else:
                    for key in  acts.keys():
                        avg_act_diffs[key] += acts_diff[key]
            
            #breakpoint()
    else:
        for line in tqdm(data):
            acts = collect_acts(model, tokenizer, line, module_type, activation_method)
            acts_diff = torch.abs(acts.sum(dim=-2)) if module_type == 'mlp' else acts
            if avg_act_diffs is None:
                avg_act_diffs = acts_diff
            else:
                if module_type == 'mlp':
                    avg_act_diffs += acts_diff
                else:
                    for key in avg_act_diffs.keys():
                        avg_act_diffs[key] += acts[key]

    if module_type == 'mlp':
        return avg_act_diffs/len(data)
    else:
        for key in avg_act_diffs.keys():
            avg_act_diffs[key]/= len(data)
            avg_act_diffs[key] = avg_act_diffs[key].numpy()
        return avg_act_diffs

def get_sensitive_neurons(data, contrasting_data, model, tokenizer, module_type, batch_size, activation_method='last'):
    avg_act_diffs = get_act_diffs_avg(data, contrasting_data, model, tokenizer, module_type, batch_size, activation_method)
    if module_type == 'mlp':
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
    else:
        
        flattened_neuron_acts = dict()
        sorted_neurons_info_per_comp = dict()
        for key in avg_act_diffs.keys():
            flattened_neuron_acts[key] = einops.rearrange(avg_act_diffs[key], 'l n_neurons -> (l n_neurons)')
            sorted_neurons = np.argsort(flattened_neuron_acts[key])[::-1]
            sorted_neurons_info = []
            neurons_per_layer_cnt = avg_act_diffs[key].shape[-1]
            for neuron in sorted_neurons:
                layer_pos = neuron//neurons_per_layer_cnt
                neuron_pos = neuron%neurons_per_layer_cnt
                neuron_name = f"L{layer_pos}N{neuron_pos}"
                sorted_neurons_info.append({
                        'name': neuron_name,
                        'layer': layer_pos.item(),
                        'neuron_pos': neuron_pos.item(),
                        'importance': flattened_neuron_acts[key][neuron]
                })
            sorted_neurons_info_per_comp[key] = sorted_neurons_info
        return sorted_neurons_info_per_comp 



def load_text_data(main_filepath: str, contrasting_filepaths: list):
    main_data = []
    with open(main_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            main_data.append(line.lower().strip())
    
    contrasting_data = []
    for contrasting_filepath in contrasting_filepaths:
       contrasting_instance = []
       with open(contrasting_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            contrasting_instance.append(line.lower().strip())
        contrasting_data.append(contrasting_instance)

    final_constrasting_data = []
    for constrasting_row in list(zip(*contrasting_data)):
        final_constrasting_data.append(list(constrasting_row))
    
    return main_data, final_constrasting_data

def main(args):
    acts = dict()
    if args.load_from_neuron_file:
        acts = load_pkl(args.neuron_file)
    
    main_data, contrasting_data = load_text_data(args.main_dataset_file, args.contrasting_dataset_files)

    batch_size = args.batch_size
    
    for model_name in tqdm(args.model_names):
        model_name_suffix = model_name.rsplit('/', 1)[-1]
        if model_name_suffix is acts:
            continue
        model, tokenizer = initialize_model_and_tokenizer(model_name)
        model_acts = get_sensitive_neurons(main_data, contrasting_data, model, tokenizer, args.module_type, batch_size, activation_method='last')
        
        acts[model_name_suffix] = model_acts
        save_pkl(acts, args.neuron_file)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_file', type=str)
    parser.add_argument('--load_from_neuron_file', action='store_true')
    parser.add_argument('--main_dataset_file', type=str)
    parser.add_argument('--contrasting_dataset_files', type=str, nargs='+', default=[])
    parser.add_argument('--model_names', type=str, nargs='+')
    parser.add_argument('--module_type', choices=('attention', 'mlp'))
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    seed_everything(42)
    main(args)
