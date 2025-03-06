from hooks import MLPActivationRecorder, MLPGradRecorder
from attention_hooks import BloomSelfAttentionWrapper, SelfAttentionWrapper
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import argparse
import einops
from utils import load_neuron_acts, save_neuron_acts, initialize_model_and_tokenizer, seed_everything
import gc
import numpy as np
from losses import SimilarityContrastiveLoss

def get_acts_and_sentence_embeddings(model, tokenizer, sentences_batch, activation_method='last'):
    model.eval()
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

    outputs = model(**tokenized_sentences, output_hidden_states=True)
    outputs = outputs.hidden_states[-1][:, -1, :]


    for act_recorder in act_recorders:
        act_recorder.remove_hook()
        
    acts = torch.cat([act_recorder.layer_outputs[0].mean(dim=0).unsqueeze(0) for act_recorder in act_recorders], dim=0)
    return acts.detach().cpu().float().numpy(), outputs


def get_sensitive_neurons(data, parallel_data, model, tokenizer, batch_size, activation_method='last'):
    loss_fn = SimilarityContrastiveLoss()
    for start_idx in tqdm(range(0, len(data), batch_size)):
        main_batch = data[start_idx:min(len(data), start_idx+batch_size)]
        parallel_batch = parallel_data[start_idx:min(len(data), start_idx+batch_size)]
        main_acts, main_embeddings = get_acts_and_sentence_embeddings(model, tokenizer, main_batch)
        parallel_acts, parallel_embeddings = get_acts_and_sentence_embeddings(model, tokenizer, parallel_batch)

        if 'LlamaForCausalLM' in str(type(model)) or 'CohereForCausalLM' in str(type(model)) or 'Qwen2ForCausalLM' in str(type(model)):
            grad_recorders = [MLPGradRecorder(layer.mlp.up_proj.weight) for layer in model.model.layers]
        elif 'BloomForCausalLM' in str(type(model)):
            grad_recorders = [MLPGradRecorder(layer.mlp.up_proj.weight) for layer in model.transformer.h]
        elif 'JAISLMHeadModel' in str(type(model)):
            grad_recorders = [MLPGradRecorder(layer.mlp.up_proj.weight) for layer in model.transformer.h]
        else:
            act_recorders = []
            print('model is not supported!')

        batch_loss = loss_fn(main_embeddings, parallel_embeddings)
        batch_loss.backward()





def load_text_data(main_filepath: str):
    main_data = []
    with open(main_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            main_data.append(line.lower().strip())
    
    
    return main_data

def main(args):
    acts = dict()
    if args.load_from_neuron_file:
        acts = load_neuron_acts(args.neuron_file)

    main_data, parallel_data = load_text_data(args.parallel_dataset_files[0]), load_text_data(args.parallel_dataset_files[1])
    

    batch_size = args.batch_size
    
    for model_name in tqdm(args.model_names):
        model_name_suffix = model_name.rsplit('/', 1)[-1]
        if model_name_suffix is acts:
            continue
        model, tokenizer = initialize_model_and_tokenizer(model_name)
        model_acts = get_sensitive_neurons(main_data, parallel_data, model, tokenizer, batch_size, activation_method='last')
        
        acts[model_name_suffix] = model_acts
        save_neuron_acts(acts, args.neuron_file)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_file', type=str)
    parser.add_argument('--load_from_neuron_file', action='store_true')
    parser.add_argument('--parallel_dataset_files', type=str, nargs='+')
    parser.add_argument('--model_names', type=str, nargs='+')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    seed_everything(42)
    main(args)
