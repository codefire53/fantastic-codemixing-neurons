from hooks import Deactivator
from utils import initialize_model_and_tokenizer, seed_everything, load_neuron_acts, save_neuron_acts
import argparse
import torch
import os
import numpy as np


import torch.nn.functional as F
from tqdm import tqdm
import einops
import numpy as np


prompt = '''Please produce the requested sentence based on the given instruction and do not add any extra text. Please write it only in the latin script.
Assuming that you are an {language_pair} bilingual speaker,
how would you write a natural {language_pair} code-mixed
sentence about {topic}? '''

topics = ['food', 'family', 'traffic', 'weather']

def generate_text(prompt, model, tokenizer, args):
    prompt_order = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(prompt_order, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts[0]


def main(args):
    if os.path.exists(args.output_file):
        print("File exists")
        return
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    model.eval()
    all_responses = []
    with torch.no_grad():
        for instance_idx in tqdm(range(args.num_of_examples)):
            for topic in topics:
                complete_prompt = prompt.format(language_pair=args.language, topic=topic)
                response = generate_text(complete_prompt, model, tokenizer, args)
                print(response)
                all_responses.append(response)
                with open(args.output_file, 'w') as f:
                    f.write('<eos>\n\n'.join(all_responses))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--language', type=str)
    parser.add_argument('--num_of_examples', type=int, default=250)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    args = parser.parse_args()
    main(args)

