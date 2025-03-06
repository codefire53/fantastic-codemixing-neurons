import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import argparse
import pickle
pio.kaleido.scope.mathjax = None

def jaccard_similarity(A, B):
    A = set(A)
    B = set(B)
    common_set = A.intersection(B)
    union_set = A.union(B)
    return len(common_set)/len(union_set)



def jaccard_similarity_neurons(top_neurons, neuron_names, model_name, top_k, out_file):
    per_model_sensitive_neurons = []
    
    # single concept
    for instance in top_neurons:
        top_neuron_infos = instance[model_name][:top_k]
        top_neuron_names = [neuron_info["name"] for neuron_info in top_neuron_infos]
        per_model_sensitive_neurons.append(top_neuron_names)
    
    
    x_labels = neuron_names
    y_labels = neuron_names
    
    similarity = []
    for y_pos in range(len(per_model_sensitive_neurons)):
        row = []
        for x_pos in  range(len(per_model_sensitive_neurons)):
            row.append(jaccard_similarity(per_model_sensitive_neurons[y_pos], per_model_sensitive_neurons[x_pos]))
        similarity.append(row)
    #breakpoint()
    fig = px.imshow(similarity,
                labels=dict(x="Concept", y="Concept", color="Overlap"),
                x=x_labels,
                y=y_labels,text_auto=True,color_continuous_scale='Inferno'
               )

    fig.update_xaxes(side="top")
    pio.full_figure_for_development(fig, warn=False)
    fig.write_image(out_file,engine="kaleido")
    

def read_pkl_file(filepath: str):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_files', nargs='+', type=str)
    parser.add_argument('--concept_names', nargs='+', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--top_k', type=int, default=2500)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()

    acts = []
    for act_file in args.activation_files:
        act = read_pkl_file(act_file)
        acts.append(act)
    jaccard_similarity_neurons(acts, args.concept_names, args.model_name, args.top_k, args.out_file)