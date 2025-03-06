#!/bin/bash

#SBATCH --job-name=syntactic_parse_graph # Job name
#SBATCH --output=/home/mahardika.ihsani/cmix_neurons/multilingual-probing-visualization/logs/output_.%A.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=40G # Total RAM to be used
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH --gres=gpu:1 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs


echo "generate syntactic parse..."
echo "aya"
cd scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-baseline_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-baseline_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-baseline_hi.yaml
rm ./embeddings/aya/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix-lang-en-hi_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-nat-cmix-lang-en-hi_hi.yaml
rm ./embeddings/aya/calcs.hdf5  


cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-nat-cmix_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-nat-cmix_hi.yaml
rm ./embeddings/aya/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.25-cmix_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.25-cmix_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.5-cmix_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.5-cmix_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-synth-0.75-cmix_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../aya-expanse-8b_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-synth-0.75-cmix_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-1_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-1_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-1_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-2_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-2_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-2_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-3_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-3_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-3_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-4_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-4_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-4_hi.yaml
rm ./embeddings/aya/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en-hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-5_natural_cmix.yaml
rm ./embeddings/aya/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b en --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/aya/en-aya-calcs-random-5_en.yaml
rm ./embeddings/aya/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/aya/calcs.hdf5 CohereForAI/aya-expanse-8b hi --neuron_file ../../random-neurons-aya-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/aya/hi-aya-calcs-random-5_hi.yaml
rm ./embeddings/aya/calcs.hdf5

echo "bloom"
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-baseline_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-baseline_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-baseline_hi.yaml
rm ./embeddings/bloom/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix-lang-en-hi_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-nat-cmix-lang-en-hi_hi.yaml
rm ./embeddings/bloom/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-nat-cmix_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-nat-cmix_hi.yaml
rm ./embeddings/bloom/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.25-cmix_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.25-cmix_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.5-cmix_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.5-cmix_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../bloom-7b1_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-synth-0.75-cmix_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../bloom-7b1_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-synth-0.75-cmix_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-1_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-1_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-1_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-2_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-2_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-2_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-3_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-3_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-3_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-4_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-4_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-4_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en-hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-5_natural_cmix.yaml
rm ./embeddings/bloom/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 en --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/bloom/en-bloom-calcs-random-5_en.yaml
rm ./embeddings/bloom/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/bloom/calcs.hdf5 bigscience/bloom-7b1 hi --neuron_file ../../random-neurons-bloom-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/bloom/hi-bloom-calcs-random-5_hi.yaml
rm ./embeddings/bloom/calcs.hdf5

echo "Llama3"
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-baseline_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-baseline_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-baseline_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix-lang-en-hi_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix-lang-en-hi_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-nat-cmix_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-nat-cmix_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.25-cmix_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.25-cmix_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.5-cmix_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.5-cmix_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-synth-0.75-cmix_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../Llama-3.2-3B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-synth-0.75-cmix_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-1_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-1_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-1_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-2_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-2_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-2_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-3_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-3_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-3_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-4_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-4_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-4_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en-hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-5_natural_cmix.yaml
rm ./embeddings/llama3.2/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B en --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/llama3.2/en-llama3.2-calcs-random-5_en.yaml
rm ./embeddings/llama3.2/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/llama3.2/calcs.hdf5 meta-llama/Llama-3.2-3B hi --neuron_file ../../random-neurons-llama3-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/llama3.2/hi-llama3.2-calcs-random-5_hi.yaml
rm ./embeddings/llama3.2/calcs.hdf5

echo "qwen"
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-baseline_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-baseline_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-baseline_hi.yaml
rm ./embeddings/qwen/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix-lang-en-hi_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_natural_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-nat-cmix-lang-en-hi_hi.yaml
rm ./embeddings/qwen/calcs.hdf5  

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-nat-cmix_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_natural_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-nat-cmix_hi.yaml
rm ./embeddings/qwen/calcs.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix-lang-en-hi_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.25-cmix-lang-en-hi_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.25-cmix_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.25_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.25-cmix_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix-lang-en-hi_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.5-cmix-lang-en-hi_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.5-cmix_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.5_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.5-cmix_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix-lang-en-hi_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix-lang-en-hi_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing_lang_en-hi.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.75-cmix-lang-en-hi_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-synth-0.75-cmix_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../Qwen2.5-7B_synth_0.75_codemixing.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-synth-0.75-cmix_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-1_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-1_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-1_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-2_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-2_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-2_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-3_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-3_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-3_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-4_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-4_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-4_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_natural_cmix.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en-hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-5_natural_cmix.yaml
rm ./embeddings/qwen/calcs.hdf5
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_en.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B en --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/en/qwen/en-qwen-calcs-random-5_en.yaml
rm ./embeddings/qwen/calcs.hdf5  
cd ../scripts
python convert_raw_to_gpt.py ../datasets/calcs_english-hinglish/dev_hi.txt ./embeddings/qwen/calcs.hdf5 Qwen/Qwen2.5-7B hi --neuron_file ../../random-neurons-qwen-1.pkl
cd ../probing
python run_inference.py ../acl2020/inference/hi/qwen/hi-qwen-calcs-random-5_hi.yaml
rm ./embeddings/qwen/calcs.hdf5

echo "Finished"