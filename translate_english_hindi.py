from tqdm import tqdm
from googletrans import Translator

def translate_from_english_to_hindi(sentences, translator, transliterator=None):
    translated_sents = []
    for sentence in tqdm(sentences):
        hindi_dev_sent = translator.translate(sentence,dest='hi',src='en')
        #hindi_latin_sent = transliterator.translit_sentence(hindi_dev_sent.text, lang_code='hi')
        translated_sents.append(hindi_dev_sent)
    return translated_sents


if __name__ == '__main__':
    translator = Translator()
    #transliterator = XlitEngine(src_script_type="indic", beam_width=3, rescore=False)
    #Calcs preprocessing
    english_data, cm_data = [], []
    with open('datasets/calcs_english-hinglish/train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            english_text, cm_text = line.split('\t')
            english_text, cm_text = english_text.strip(), cm_text.strip()
            english_data.append(english_text)
            cm_data.append(cm_text)


        hindi_data = translate_from_english_to_hindi(english_data, translator, transliterator)

        lines = ['\t'.join([en_sent, hi_sent, cm_sent]) for en_sent, hi_sent, cm_sent in zip(english_data, hindi_data, cm_data)]

    with open('datasets/calcs_english-hinglish/train_hi_og.txt', 'w') as f:
        f.write('\n'.join(lines))
    
    english_data, cm_data = [], []

    with open('datasets/calcs_english-hinglish/dev.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            english_text, cm_text = line.split('\t')
            english_text, cm_text = english_text.strip(), cm_text.strip()
            english_data.append(english_text)
            cm_data.append(cm_text)


    hindi_data = translate_from_english_to_hindi(english_data, translator, transliterator)

    lines = ['\t'.join([en_sent, hi_sent, cm_sent]) for en_sent, hi_sent, cm_sent in zip(english_data, hindi_data, cm_data)]

    with open('datasets/calcs_english-hinglish/dev_hi_og.txt', 'w') as f:
        f.write('\n'.join(lines))


    english_data, cm_data = [], []

    with open('datasets/calcs_english-hinglish/test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            english_text, cm_text = line.split('\t')
            english_text, cm_text = english_text.strip(), cm_text.strip()
            english_data.append(english_text)
            cm_data.append(cm_text)


    hindi_data = translate_from_english_to_hindi(english_data, translator, transliterator)

    lines = ['\t'.join([en_sent, hi_sent, cm_sent]) for en_sent, hi_sent, cm_sent in zip(english_data, hindi_data, cm_data)]

    with open('datasets/calcs_english-hinglish/test_hi_og.txt', 'w') as f:
        f.write('\n'.join(lines))