#!/bin/bash

#SBATCH --job-name=generate-synth-cmix # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l4-003


echo "Generating synthetic codemixed"
python generate_synth_cmix.py --in_dataset_file datasets/spanglish/train_all_sentences.txt --out_natural_cm_dataset_file datasets/spanglish/train_natural_cmix.txt  --out_synth_cm_dataset_file datasets/spanglish/train_synth_cmix_proba_0.75.txt --out_l1_dataset_file datasets/spanglish/train_en_sentences.txt --out_l2_dataset_file datasets/spanglish/train_es_sentences.txt --switch_proba 0.75
python generate_synth_cmix.py --in_dataset_file datasets/spanglish/train_all_sentences.txt --out_natural_cm_dataset_file datasets/spanglish/train_natural_cmix.txt  --out_synth_cm_dataset_file datasets/spanglish/train_synth_cmix_proba_0.5.txt --out_l1_dataset_file datasets/spanglish/train_en_sentences.txt --out_l2_dataset_file datasets/spanglish/train_es_sentences.txt --switch_proba 0.5
python generate_synth_cmix.py --in_dataset_file datasets/spanglish/train_all_sentences.txt --out_natural_cm_dataset_file datasets/spanglish/train_natural_cmix.txt  --out_synth_cm_dataset_file datasets/spanglish/train_synth_cmix_proba_0.25.txt --out_l1_dataset_file datasets/spanglish/train_en_sentences.txt --out_l2_dataset_file datasets/spanglish/train_es_sentences.txt --switch_proba 0.25

 
