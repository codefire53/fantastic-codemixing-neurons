# Syntactic Neurons Analysis For Codemixing
This part actually follows https://github.com/ethanachi/multilingual-probing-visualization repo, with few adjustments for LLMs and enabling the intervention.

## Setup
For installation and also for specification of yaml files for training and inference, please refer to `README_original_repo.md`. The configurations for training are put inside of `acl2020/direct` folder and for inference inside of `acl2020/inference` folder.


## Train
Once you have specified the configuration you can follow similar command as in `train_multilingual_structural_probe.sh` consisting of `convert_raw_to_gpt.py` to get all the embedding features and `run_experiment.py` to run the training

## Inference
For inference is quite similar but instead of running `run_experiment.py` you have to run `run_inference.py`, you can see `generate_syntactic_parse_graph.sh` as your reference. Note that  there is `neuron_file` optional argument that you have to provide in `convert_raw_to_gpt.py` to get representations from model that is intervened. This inference will produce approximated syntax graph for given sentences. For now we store all these graphs into a folder called `edges`. To analyze the graph edit distance, you can use `https://github.com/jajupmochi/graphkit-learn`. 