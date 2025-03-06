import plotly.graph_objects as go
from collections import Counter
from plotly.figure_factory import create_distplot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

def plot_sensitive_neuron_distributions(neuron_acts_file, concept, top_k, out_file):
    with open(neuron_acts_file, 'rb') as f:
        sorted_neurons_info_per_model = pickle.load(f)
    hist_data = []
    group = []
    for model_name, sorted_neurons_info in sorted_neurons_info_per_model.items():
        sensitive_neurons = sorted_neurons_info[:top_k]
        max_layer = max([instance['layer'] for instance in sorted_neurons_info])
        sensitive_neuron_layers = [instance['layer']/max_layer for instance in sensitive_neurons]
        hist_data.extend(sensitive_neuron_layers)
        group.extend([model_name]*(len(sensitive_neuron_layers)))
    df = pd.DataFrame.from_dict({
        'Relative Layer': hist_data, 'Model': group
    })
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    sns.kdeplot(df, x='Relative Layer', hue='Model')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.title(f"Top-{top_k} neurons distribution of {concept} concept")
    plt.savefig(args.out_file, format="pdf", bbox_inches="tight")
    plt.show() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_name', type=str)
    parser.add_argument('--activation_file', type=str)
    parser.add_argument('--top_k', type=int, default=2500)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()
    plot_sensitive_neuron_distributions(args.activation_file, args.concept_name, args.top_k, args.out_file)