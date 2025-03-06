from evaluate import load
import argparse

def load_txt_file(filepath):
    lines = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip())
    return lines

def load_txt_file_with_sep(filepath,sep='|'):
    lines = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            text = line.split(sep)[0]
            lines.append(text.strip())
    return lines
def eval_text_quality(args):
    source_texts = load_txt_file(args.src_file)
    ref_texts = load_txt_file(args.codemixed_ref_file)
    pred_texts = load_txt_file_with_sep(args.pred_file)
    ref_texts_2 = load_txt_file(args.hindi_ref_file)
    bleu_score = sacrebleu_metric.compute(references=ref_texts, predictions=pred_texts)
    bleu_score_2 = sacrebleu_metric.compute(references=ref_texts_2, predictions=pred_texts)
    comet_score = comet_metric.compute(predictions=pred_texts, references=ref_texts, sources=source_texts)
    comet_score_2 = comet_metric.compute(predictions=pred_texts, references=ref_texts_2, sources=source_texts)
    print(f"BLEU Score: {max(bleu_score['score'],bleu_score_2['score'])}")
    print(f"Comet Score: {max(comet_score['mean_score'], comet_score_2['mean_score'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--codemixed_ref_file', type=str)
    parser.add_argument('--hindi_ref_file', type=str)
    parser.add_argument('--pred_file', type=str)

    args = parser.parse_args()
    eval_text_quality(args)