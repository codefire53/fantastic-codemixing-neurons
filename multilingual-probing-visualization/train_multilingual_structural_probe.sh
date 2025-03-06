#!/bin/bash

#SBATCH --job-name=structural_probe_training # Job name
#SBATCH --output=/home/mahardika.ihsani/cmix_neurons/multilingual-probing-visualization/logs/output_.%A.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=40G # Total RAM to be used
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH --gres=gpu:1 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs


echo "train hindi structural probe..."
# aya
cd scripts
python convert_raw_to_gpt.py ../datasets/hi/train_roman.txt ./embeddings/aya/train_hi.hdf5 CohereForAI/aya-expanse-8b hi
python convert_raw_to_gpt.py ../datasets/hi/dev_roman.txt ./embeddings/aya/dev_hi.hdf5 CohereForAI/aya-expanse-8b hi
python convert_raw_to_gpt.py ../datasets/hi/test_roman.txt ./embeddings/aya/test_hi.hdf5 CohereForAI/aya-expanse-8b hi
cd ../probing
python run_experiment.py ../acl2020/direct/hi-aya-31-128.yaml
rm ./embeddings/aya/train_hi.hdf5 
rm ./embeddings/aya/dev_hi.hdf5 
rm ./embeddings/aya/test_hi.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/hi/train_roman.txt ./embeddings/bloom/train_hi.hdf5 bigscience/bloom-7b1 hi
python convert_raw_to_gpt.py ../datasets/hi/dev_roman.txt ./embeddings/bloom/dev_hi.hdf5 bigscience/bloom-7b1 hi
python convert_raw_to_gpt.py ../datasets/hi/test_roman.txt ./embeddings/bloom/test_hi.hdf5 bigscience/bloom-7b1 hi
cd ../probing
python run_experiment.py ../acl2020/direct/hi-bloom-29-128.yaml
rm ./embeddings/bloom/train_hi.hdf5 
rm ./embeddings/bloom/dev_hi.hdf5 
rm ./embeddings/bloom/test_hi.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/hi/train_roman.txt ./embeddings/llama3.2/train_hi.hdf5 meta-llama/Llama-3.2-3B hi
python convert_raw_to_gpt.py ../datasets/hi/dev_roman.txt ./embeddings/llama3.2/dev_hi.hdf5 meta-llama/Llama-3.2-3B hi
python convert_raw_to_gpt.py ../datasets/hi/test_roman.txt ./embeddings/llama3.2/test_hi.hdf5 meta-llama/Llama-3.2-3B hi
cd ../probing
python run_experiment.py ../acl2020/direct/hi-llama3-27-128.yaml
rm ./embeddings/llama3.2/train_hi.hdf5 
rm ./embeddings/llama3.2/dev_hi.hdf5 
rm ./embeddings/llama3.2/test_hi.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/hi/train_roman.txt ./embeddings/qwen/train_hi.hdf5 Qwen/Qwen2.5-7B hi
python convert_raw_to_gpt.py ../datasets/hi/dev_roman.txt ./embeddings/qwen/dev_hi.hdf5 Qwen/Qwen2.5-7B hi
python convert_raw_to_gpt.py ../datasets/hi/test_roman.txt ./embeddings/qwen/test_hi.hdf5 Qwen/Qwen2.5-7B hi
cd ../probing
python run_experiment.py ../acl2020/direct/hi-qwen-27-128.yaml
rm ./embeddings/qwen/train_hi.hdf5 
rm ./embeddings/qwen/dev_hi.hdf5 
rm ./embeddings/qwen/test_hi.hdf5

echo "train english structural probe..."
cd ../scripts
python convert_raw_to_gpt.py ../datasets/en/train.txt ./embeddings/aya/train_en.hdf5 CohereForAI/aya-expanse-8b en
python convert_raw_to_gpt.py ../datasets/en/dev.txt ./embeddings/aya/dev_en.hdf5 CohereForAI/aya-expanse-8b en
python convert_raw_to_gpt.py ../datasets/en/test.txt ./embeddings/aya/test_en.hdf5 CohereForAI/aya-expanse-8b en
cd ../probing
python run_experiment.py ../acl2020/direct/en-aya-31-128.yaml
rm ./embeddings/aya/train_en.hdf5 
rm ./embeddings/aya/dev_en.hdf5 
rm ./embeddings/aya/test_en.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/en/train.txt ./embeddings/bloom/train_en.hdf5 bigscience/bloom-7b1 en
python convert_raw_to_gpt.py ../datasets/en/dev.txt ./embeddings/bloom/dev_en.hdf5 bigscience/bloom-7b1 en
python convert_raw_to_gpt.py ../datasets/en/test.txt ./embeddings/bloom/test_en.hdf5 bigscience/bloom-7b1 en
cd ../probing
python run_experiment.py ../acl2020/direct/en-bloom-29-128.yaml
rm ./embeddings/bloom/train_en.hdf5 
rm ./embeddings/bloom/dev_en.hdf5 
rm ./embeddings/bloom/test_en.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/en/train.txt ./embeddings/llama3.2/train_en.hdf5 meta-llama/Llama-3.2-3B en
python convert_raw_to_gpt.py ../datasets/en/dev.txt ./embeddings/llama3.2/dev_en.hdf5 meta-llama/Llama-3.2-3B en
python convert_raw_to_gpt.py ../datasets/en/test.txt ./embeddings/llama3.2/test_en.hdf5 meta-llama/Llama-3.2-3B en
cd ../probing
python run_experiment.py ../acl2020/direct/en-llama3-27-128.yaml
rm ./embeddings/llama3.2/train_en.hdf5 
rm ./embeddings/llama3.2/dev_en.hdf5 
rm ./embeddings/llama3.2/test_en.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/en/train.txt ./embeddings/qwen/train_en.hdf5 Qwen/Qwen2.5-7B en
python convert_raw_to_gpt.py ../datasets/en/dev.txt ./embeddings/qwen/dev_en.hdf5 Qwen/Qwen2.5-7B en
python convert_raw_to_gpt.py ../datasets/en/test.txt ./embeddings/qwen/test_en.hdf5 Qwen/Qwen2.5-7B en
cd ../probing
python run_experiment.py ../acl2020/direct/en-qwen-27-128.yaml
rm ./embeddings/qwen/train_en.hdf5 
rm ./embeddings/qwen/dev_en.hdf5 
rm ./embeddings/qwen/test_en.hdf5

echo "train arabic structural probe..."
cd ../scripts
python convert_raw_to_gpt.py ../datasets/ar/train.txt ./embeddings/aya/train_ar.hdf5 CohereForAI/aya-expanse-8b ar
python convert_raw_to_gpt.py ../datasets/ar/dev.txt ./embeddings/aya/dev_ar.hdf5 CohereForAI/aya-expanse-8b ar
python convert_raw_to_gpt.py ../datasets/ar/test.txt ./embeddings/aya/test_ar.hdf5 CohereForAI/aya-expanse-8b ar
cd ../probing
python run_experiment.py ../acl2020/direct/ar-aya-31-128.yaml
rm ./embeddings/aya/train_ar.hdf5 
rm ./embeddings/aya/dev_ar.hdf5 
rm ./embeddings/aya/test_ar.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/ar/train.txt ./embeddings/bloom/train_ar.hdf5 bigscience/bloom-7b1 ar
python convert_raw_to_gpt.py ../datasets/ar/dev.txt ./embeddings/bloom/dev_ar.hdf5 bigscience/bloom-7b1 ar
python convert_raw_to_gpt.py ../datasets/ar/test.txt ./embeddings/bloom/test_ar.hdf5 bigscience/bloom-7b1 ar
cd ../probing
python run_experiment.py ../acl2020/direct/ar-bloom-29-128.yaml
rm ./embeddings/bloom/train_ar.hdf5 
rm ./embeddings/bloom/dev_ar.hdf5 
rm ./embeddings/bloom/test_ar.hdf5 

cd ../scripts
python convert_raw_to_gpt.py ../datasets/ar/train.txt ./embeddings/qwen/train_ar.hdf5 Qwen/Qwen2.5-7B ar
python convert_raw_to_gpt.py ../datasets/ar/dev.txt ./embeddings/qwen/dev_ar.hdf5 Qwen/Qwen2.5-7B ar
python convert_raw_to_gpt.py ../datasets/ar/test.txt ./embeddings/qwen/test_ar.hdf5 Qwen/Qwen2.5-7B ar
cd ../probing
python run_experiment.py ../acl2020/direct/ar-qwen-27-128.yaml
rm ./embeddings/qwen/train_ar.hdf5 
rm ./embeddings/qwen/dev_ar.hdf5 
rm ./embeddings/qwen/test_ar.hdf5

echo "Finished"