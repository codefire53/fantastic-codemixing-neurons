import argparse
from simalign import SentenceAligner
from nltk.tokenize import word_tokenize
import random
from tqdm import tqdm

def generate_synth_codemixing(aligner, src_sent, tgt_sent, switch_proba):
    alignment_method = 'itermax'
    src_words = word_tokenize(src_sent)
    tgt_words = word_tokenize(tgt_sent)
    alignments = aligner.get_word_aligns(src_words, tgt_words)
    alignments = alignments[alignment_method]
    alignment_maps = dict()

    for mapping_tuple in alignments:
        src_idx, tgt_idx = mapping_tuple[0], mapping_tuple[1]
        if src_idx not in alignment_maps:
            alignment_maps[src_idx] = []
        alignment_maps[src_idx].append(tgt_idx)
    
    cmix_words = src_words.copy()
    for idx, word in enumerate(src_words):
        if idx in alignment_maps:
            is_switch = bool(random.choices([0, 1], [1-switch_proba, switch_proba])[0])
            if is_switch:
                aligned_words = []
                for tgt_idx in alignment_maps[idx]:
                    aligned_words.append(tgt_words[tgt_idx])
                aligned_words = " ".join(aligned_words)
                cmix_words[idx] = aligned_words
    return " ".join(cmix_words).lower().strip()
            

def write_to_file(sents, out_file):
    with open(out_file, 'w') as f:
        all_sents = '\n'.join(sents)
        f.write(all_sents)


def main(args):
    aligner = SentenceAligner()
    l1_sents = []
    l2_sents = []
    cm_sents = []
    synth_cm_sents = []
    with open(args.in_dataset_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            l1_sent, l2_sent, cm_sent = line.split('\t')
            l1_sent = l1_sent.lower().strip()
            l2_sent = l2_sent.lower().strip()
            cm_sent = cm_sent.lower().strip()
            if l1_sent.strip() == '' or l2_sent.strip() == '':
                continue
            synth_cm_sent = generate_synth_codemixing(aligner, l1_sent, l2_sent, args.switch_proba)
            l1_sents.append(l1_sent)
            l2_sents.append(l2_sent)
            cm_sents.append(cm_sent)
            synth_cm_sents.append(synth_cm_sent)


    write_to_file(l1_sents, args.out_l1_dataset_file)
    write_to_file(l2_sents, args.out_l2_dataset_file)
    write_to_file(cm_sents, args.out_natural_cm_dataset_file)
    write_to_file(synth_cm_sents, args.out_synth_cm_dataset_file)

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dataset_file', type=str)
    parser.add_argument('--out_natural_cm_dataset_file', type=str)
    parser.add_argument('--out_synth_cm_dataset_file', type=str)
    parser.add_argument('--out_l1_dataset_file', type=str)
    parser.add_argument('--out_l2_dataset_file', type=str)
    parser.add_argument('--switch_proba', type=float)
    args = parser.parse_args()
    main(args)
