#!/bin/bash

#SBATCH --job-name=generate-llm-cmix # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l4-021


echo "Generating synthetic codemixed"
python generate_codeswitching_sentences.py --model_name CohereForAI/aya-expanse-8b --language English-Hindi --output_file datasets/model-generated-texts/aya/hinglish.txt --num_of_examples 250
python generate_codeswitching_sentences.py --model_name bigscience/bloomz-7b1 --language English-Hindi --output_file datasets/model-generated-texts/bloom-7b1/hinglish.txt --num_of_examples 250
python generate_codeswitching_sentences.py --model_name meta-llama/Llama-3.2-3B-Instruct --language English-Hindi --output_file datasets/model-generated-texts/Llama3.2-3B/hinglish.txt --num_of_examples 250
python generate_codeswitching_sentences.py --model_name Qwen/Qwen2.5-7B-Instruct --language English-Hindi --output_file datasets/model-generated-texts/qwen-7b/hinglish.txt --num_of_examples 250


 
