import math
import torch

class MLPActivationRecorder:
    def __init__(self, target_mlp_layer, method:str = 'last'):
        self.target_mlp_layer = target_mlp_layer
        self.layer_outputs = []
        if method == 'avg':
            self.save_hook = target_mlp_layer.register_forward_hook(self.save_activations_avg_hook)
        else:
            self.save_hook = target_mlp_layer.register_forward_hook(self.save_activations_hook)
    
    def save_activations_hook(self, module, input, output):
        self.layer_outputs.append(output[:,-1,:].detach().cpu())
    
    def save_activations_avg_hook(self, module, input, output):
        self.layer_outputs.append(output.mean(dim=1).detach().cpu())
    
    def remove_hook(self):
        self.save_hook.remove()

class MLPSwitchActivationRecorder:
    def __init__(self, target_mlp_layer, switch_point_positions_per_batch: list, non_switch_point_positions_per_batch: list = [], method:str = 'contrast'):
        self.target_mlp_layer = target_mlp_layer
        self.layer_switch_point_acts = []
        self.layer_nonswitch_point_acts = []
        if method == 'contrast':
            self.save_hook = target_mlp_layer.register_forward_hook(self.save_contrasting_activations_hook)
        else:
            self.save_hook = target_mlp_layer.register_forward_hook(self.save_switch_point_activations_hook)
        self.switch_point_positions_per_batch = switch_point_positions_per_batch
        self.non_switch_point_positions_per_batch = non_switch_point_positions_per_batch
    
    def save_contrasting_activations_hook(self, module, input, output):
        batch_size = len(self.switch_point_positions_per_batch)
        for batch_pos in range(batch_size):
            acts = output[batch_pos].detach().cpu().float()
            switch_positions = self.switch_point_positions_per_batch[batch_pos]
            nonswitch_positions = self.non_switch_point_positions_per_batch[batch_pos]
            self.layer_switch_point_acts.append(acts[switch_positions,:].sum(dim=0).unsqueeze(0))
            self.layer_nonswitch_point_acts.append(acts[nonswitch_positions,:].sum(dim=0).unsqueeze(0))
        self.layer_switch_point_acts = torch.cat(self.layer_switch_point_acts, dim=0).sum(dim=0)
        self.layer_nonswitch_point_acts = torch.cat(self.layer_nonswitch_point_acts, dim=0).sum(dim=0)
        
    def save_switch_point_activations_hook(self, module, input, output):
        batch_size = len(self.switch_point_positions_per_batch)
        for batch_pos in range(batch_size):
            acts = output[batch_pos].detach().cpu().float()
            switch_positions = self.switch_point_positions_per_batch[batch_pos]
            self.layer_switch_point_acts.append(acts[switch_positions,:].sum(dim=0).unsqueeze(0))
        self.layer_switch_point_acts = torch.cat(self.layer_switch_point_acts, dim=0).sum(dim=0)
        self.layer_nonswitch_point_acts = torch.zeros_like(self.layer_switch_point_acts, device=self.layer_switch_point_acts.device)
    
    def remove_hook(self):
        self.save_hook.remove()

class MLPGradRecorder:
    def __init__(self, target_mlp_weight):
        self.target_mlp_weight = target_mlp_weight
        self.layer_grads = None
        self.save_hook = target_mlp_weight.register_hook(self.save_grad_hook)

    
    def save_grad_hook(self, grad):
        self.layer_grads = grad.sum(axis=0).detach().cpu()
    
    def remove_hook(self):
        self.save_hook.remove()

class Deactivator:
    def __init__(self, target_mlp_layer, selected_neurons, device, patch_neuron_start_indices=[], ablation_method='zero', patching_values_map={}):
        self.target_mlp_layer = target_mlp_layer
        self.selected_neurons = selected_neurons
        self.ablation_method = ablation_method
        self.ablation_values = []
        self.patch_neuron_start_indices = patch_neuron_start_indices
        self.device = device
        
        if 'zero' not in self.ablation_method:
            for neuron_idx in selected_neurons:
                self.ablation_values.append(float(patching_values_map[neuron_idx]))
            self.ablation_values = torch.tensor(self.ablation_values, device=self.device)
        self.deactivation_hook = target_mlp_layer.register_forward_hook(self.deactivate)
    
    
    def deactivate(self, module, input, output):
        if len(self.ablation_values)==0:
            if self.ablation_method == 'zero':
                if len(self.patch_neuron_start_indices) == 0:
                    output[:, :, self.selected_neurons] *= 0
                else:
                    for batch_idx, start_idx in enumerate(self.patch_neuron_start_indices):
                        output[batch_idx,start_idx:,self.selected_neurons] *= 0
            else:
                output[:, -1, self.selected_neurons] *= 0
        
        # use mean ablation
        else:
            output[:, -1, self.selected_neurons] = self.ablation_values.to(dtype=output.dtype) # use last token since the mean information wrt the last token
        return output
    
    def remove_hook(self):
        self.deactivation_hook.remove()


class Activator:
    def __init__(self, target_mlp_layer, selected_neurons, patch_neuron_start_indices=[], multiplier_const=3):
        self.target_mlp_layer = target_mlp_layer
        self.selected_neurons = selected_neurons
        self.multiplier_const = multiplier_const
        self.patch_neuron_start_indices = patch_neuron_start_indices
        self.activation_hook = target_mlp_layer.register_forward_hook(self.activate)
    
    def activate(self, module, input, output):
        if len(self.patch_neuron_start_indices) == 0:
            output[:, :, self.selected_neurons] *= self.multiplier_const
        else:
            for batch_idx, start_idx in enumerate(self.patch_neuron_start_indices):
                output[batch_idx,start_idx:,self.selected_neurons] *= self.multiplier_const
        return output
    
    def remove_hook(self):
        self.activation_hook.remove()