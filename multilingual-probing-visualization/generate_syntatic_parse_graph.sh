#!/bin/bash

#SBATCH --job-name=syntatic-parse-calcs # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l4-021


echo "generate syntactic parse..."
# echo "aya"
# cd scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-baseline_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-baseline_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-baseline_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-lang-en-hi_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-lang-en-hi_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-nat-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5  


# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-nat-cmix_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.25-cmix_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.5-cmix_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.75-cmix_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-1_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-1_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-1_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-2_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-2_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-2_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-3_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-3_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-3_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-4_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-4_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-4_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-5_natural_cmix.yaml
# rm -f ../embeddings/aya/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-5_en.yaml
# rm -f ../embeddings/aya/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-5_hi.yaml
# rm -f ../embeddings/aya/calcs.hdf5

# echo "bloom"
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-baseline_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-baseline_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-baseline_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-lang-en-hi_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-lang-en-hi_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-nat-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_natural_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-nat-cmix_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.25-cmix_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.5-cmix_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.75-cmix_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-1_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-1_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_random-1_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-1_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-2_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-2_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_random-2_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-2_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-3_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-3_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_random-3_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-3_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-4_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-4_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_random-4_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-4_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-5_natural_cmix.yaml
# rm -f ../embeddings/bloom/calcs.hdf5
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-5_en.yaml
# rm -f ../embeddings/bloom/calcs.hdf5  
# cd ../scripts
# python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_random-5_attention.pkl --component_to_patch attention
# cd ../probing
# python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-5_hi.yaml
# rm -f ../embeddings/bloom/calcs.hdf5

echo "Llama3"
cd scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-baseline_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-baseline_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-baseline_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  

cd ../scripts
CUDA_LAUNCH_BLOCKING=1 python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-lang-en-hi_natural_cmix.yaml
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-lang-en-hi_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-lang-en-hi_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi_atttention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.25-cmix_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.5-cmix_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.75-cmix_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama3.2-3B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-1_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama3.2-3B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-1_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama3.2-3B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-1_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama3.2-3B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-2_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama3.2-3B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-2_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama3.2-3B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-2_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama3.2-3B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-3_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama3.2-3B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-3_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama3.2-3B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-3_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama3.2-3B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-4_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama3.2-3B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-4_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama3.2-3B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-4_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama3.2-3B_random-5_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-5_natural_cmix.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama3.2-3B_random-5_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-5_en.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama3.2-3B_random-5_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-5_hi.yaml
rm -f ../embeddings/llama3.2/calcs.hdf5

echo "qwen"
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-baseline_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-baseline_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-baseline_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-lang-en-hi_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-lang-en-hi_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-nat-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-nat-cmix_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.25-cmix_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.5-cmix_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.75-cmix_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-1_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-1_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-1_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-2_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-2_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_random-2_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-2_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-3_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-3_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_random-3_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-3_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-4_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-4_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_random-4_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-4_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-5_natural_cmix.yaml
rm -f ../embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-5_en.yaml
rm -f ../embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ../embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_random-1_attention.pkl --component_to_patch attention
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-5_hi.yaml
rm -f ../embeddings/qwen/calcs.hdf5

echo "Finished"
