import argparse
from utils import load_pkl, save_pkl


def aggregate_neuron_groups(neuron_groups, top_k):
    important_neurons_per_layer = dict()
    if len(neuron_groups) == 1:
        neuron_group = neuron_groups[0]
        important_neurons = neuron_group[:top_k]
        for neuron_info in important_neurons:
            layer = int(neuron_info['layer'])
            neuron_pos = int(neuron_info['neuron_pos'])
            if layer not in important_neurons_per_layer:
                important_neurons_per_layer[layer] = []
            important_neurons_per_layer[layer].append(neuron_pos)
    
    else:
        important_neurons_per_layer = dict()
        all_neurons = []
        should_finish = False
        for neuron_pos in range(top_k):
            for neuron_group in neuron_groups:
                if len(all_neurons) == top_k:
                    should_finish = True
                    break
                neuron_info = neuron_group[neuron_pos]
                neuron_info_layer = int(neuron_info['layer'])
                neuron_info_pos = int(neuron_info['neuron_pos'])
                neuron_info_name = neuron_info['name']
                all_neurons.append(neuron_info_name)
                if neuron_info_name not in all_neurons:
                    if neuron_info_layer not in important_neurons_per_layer:
                        important_neurons_per_layer[neuron_info_layer] = []
                    important_neurons_per_layer[neuron_info_layer].append(neuron_info_pos)
                    all_neurons.append(neuron_info_name)
            if should_finish:
                break

    for layer in important_neurons_per_layer.keys():
        important_neurons_per_layer[layer] = sorted(important_neurons_per_layer[layer])

    return important_neurons_per_layer   



def main(args):
    neuron_groups_all_models = []
    for neuron_file in args.in_neuron_files:
        acts = load_pkl(neuron_file)
        neuron_groups_all_models.append(acts)
    for model_name in neuron_groups_all_models.keys():
        neuron_groups = [neuron_group[model_name] for neuron_group in neuron_groups_all_models]
        aggregated_neuron_group = aggregate_neuron_groups(neuron_groups, args.top_k)
        save_pkl(aggregated_neuron_group, f'{args.out_neuron_file}_{model_name}.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_neuron_files', type=str, nargs='+')
    parser.add_argument('--out_neuron_file_prefix', type=str)
    parser.add_argument('--top_k', type=int, default=2500)
    args = parser.parse_args()
    main(args)
