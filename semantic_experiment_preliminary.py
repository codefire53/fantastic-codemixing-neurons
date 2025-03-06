from hooks import Deactivator
from utils import initialize_model_and_tokenizer, seed_everything, load_neuron_acts, save_neuron_acts
import argparse
import torch
import os
import numpy as np

def find_last_token_pos(mask):
    reversed_mask = mask.flip(1)
    last_indices = mask.size(1) - 1 - reversed_mask.argmax(dim=1)
    last_indices[reversed_mask.sum(1) == 0] = 0
    return last_indices

def extract_last_token_repr(hidden_state, last_token_indices):
    batch_indices = torch.arange(len(hidden_state))
    return hidden_state[batch_indices, last_token_indices].unsqueeze(1)

import torch.nn.functional as F
from tqdm import tqdm
import einops
import numpy as np

def collect_semantic_sim(model, tokenizer, inputs_cm, inputs_l1, inputs_l2, batch_size, patched_neurons_per_layer=None):
    all_cos_sims = []
    model.eval()
    assert len(inputs_cm) == len(inputs_l1) == len(inputs_l2)
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(inputs_cm), batch_size)):
            # codemixed
            text_batch_cm = inputs_cm[batch_start:min(batch_start+batch_size,len(inputs_cm))]
            tokenized_batch_cm = tokenizer(text_batch_cm, padding=True, return_tensors='pt')
            attention_mask_cm = tokenized_batch_cm['attention_mask']
            input_ids_cm = tokenized_batch_cm['input_ids'].to(model.device)
            attention_mask_cm = attention_mask_cm.to(model.device)
            last_hidden_state_cm = model(input_ids=input_ids_cm, attention_mask=attention_mask_cm, output_hidden_states=True).hidden_states[-1].detach().cpu()
            last_token_representation_cm  = last_hidden_state_cm[:, -1, :] # batch*dim
            
            # compute in-batch cosine
            query = last_token_representation_cm.clone().unsqueeze(1) # batch*1*dim
            key = last_token_representation_cm.clone() # batch*dim
            cos_sim = F.cosine_similarity(query, key, dim=-1).squeeze(1) # batch*batch
            n = cos_sim.shape[0]
            mask_matrix = torch.ones(n) - torch.eye(n)
            in_batch_negative_examples_avg = cos_sim*mask_matrix
            in_batch_negative_examples_avg = in_batch_negative_examples_avg.sum(dim=-1)
            in_batch_negative_examples_avg = in_batch_negative_examples_avg/(n-1)
            in_batch_negative_examples_avg = in_batch_negative_examples_avg.float().numpy()

            # l1
            text_batch_l1 = inputs_l1[batch_start:min(batch_start+batch_size,len(inputs_l1))]
            tokenized_batch_l1 = tokenizer(text_batch_l1, padding=True, return_tensors='pt')
            attention_mask_l1 = tokenized_batch_l1['attention_mask']
            input_ids_l1 = tokenized_batch_l1['input_ids'].to(model.device)
            attention_mask_l1 = attention_mask_l1.to(model.device)
            last_hidden_state_l1 = model(input_ids=input_ids_l1, attention_mask=attention_mask_l1, output_hidden_states=True).hidden_states[-1].detach().cpu()
            last_token_representation_l1  = last_hidden_state_l1[:, -1, :] # batch*dim


            # l2
            text_batch_l2 = inputs_l2[batch_start:min(batch_start+batch_size,len(inputs_l2))]
            tokenized_batch_l2 = tokenizer(text_batch_l2, padding=True, return_tensors='pt')
            attention_mask_l2 = tokenized_batch_l2['attention_mask']
            input_ids_l2 = tokenized_batch_l2['input_ids'].to(model.device)
            attention_mask_l2 = attention_mask_l2.to(model.device)
            last_hidden_state_l2 = model(input_ids=input_ids_l2, attention_mask=attention_mask_l2, output_hidden_states=True).hidden_states[-1].detach().cpu()
            last_token_representation_l2  = last_hidden_state_l2[:, -1, :] # batch*dim

            # cm & l1
            cm_l1_cos_sim = F.cosine_similarity(query, last_token_representation_l1.unsqueeze(1), dim=-1).squeeze(1).float().numpy()

            # cm & l2
            cm_l2_cos_sim = F.cosine_similarity(query, last_token_representation_l2.unsqueeze(1), dim=-1).squeeze(1).float().numpy()

            for cm_neg_cm, cm_l1, cm_l2 in zip(in_batch_negative_examples_avg, cm_l1_cos_sim, cm_l2_cos_sim):
                all_cos_sims.append({
                    f'neg_{args.matrix_lang}-{args.embedded_lang}': cm_neg_cm,
                    args.matrix_lang: cm_l1,
                    args.embedded_lang: cm_l2
                })
    return all_cos_sims

def load_data(filename: str):
    out_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            out_lines.append(line.lower().strip())
    return out_lines

def main(args):
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    model_name_suffix = args.model_name.rsplit('/', 1)[-1]
    inputs_cm = load_data(args.codemixed_file)
    inputs_l1 = load_data(args.matrix_lang_file)
    inputs_l2 = load_data(args.embedded_lang_file)
    all_sem_scores = collect_semantic_sim(model, tokenizer, inputs_cm, inputs_l1, inputs_l2, args.batch_size)
    save_neuron_acts(all_sem_scores, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--codemixed_file', type=str)
    parser.add_argument('--matrix_lang_file', type=str)
    parser.add_argument('--embedded_lang_file', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--matrix_lang', type=str)
    parser.add_argument('--embedded_lang', type=str)
    args = parser.parse_args()
    seed_everything(42)
    main(args)

