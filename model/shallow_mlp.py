import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from .utils import get_activation



class LinearFunctionFA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, backward_weight=None, weight_masks=None):
        ctx.save_for_backward(input, weight, bias, backward_weight, weight_masks)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, backward_weight, weight_masks = ctx.saved_tensors
        backward_weight = (backward_weight * weight_masks).to(grad_output.device)
        input = input.to(grad_output.device)
        grad_input = grad_output.mm(backward_weight) # use backward_weight here instead of weight.t()
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None

class LinearFA(nn.Module):
    def __init__(self, input_features, output_features, bias=False, fa_sparsity=0.):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # This registers the parameter with the module, making it appear in methods like `.parameters()` and `.to()`
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize random fixed backward weights
        self.backward_weight = torch.randn(output_features, input_features)
        stdv = 1. / math.sqrt(self.backward_weight.size(1))
        self.backward_weight.data.uniform_(-stdv, stdv)

        
        # Initialize forward weights and biases with the usual initialization method
        stdv_weight = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv_weight, stdv_weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.weight_masks = (torch.rand_like(self.backward_weight) > fa_sparsity).float()
        self.backward_weight = self.backward_weight * self.weight_masks


    def forward(self, input):
        return LinearFunctionFA.apply(input, self.weight, self.bias, self.backward_weight, self.weight_masks)


class L4(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(activation)
    def forward(self, x):
        x = self.activation(self.fc1(x))
        return x

        
    
class L23(nn.Module):
    def __init__(self, hidden_dim, latent_dim, td, l23_l5_fa, fa_sparsity, activation):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + td, hidden_dim)
        # self.fc1 = nn.Linear(hidden_dim *2, hidden_dim)
        
        self.activation = get_activation(activation)
        if l23_l5_fa:
            self.l23_l5 = LinearFA(hidden_dim, latent_dim, fa_sparsity=fa_sparsity)
        else:
            self.l23_l5 = nn.Linear(hidden_dim, latent_dim)
            
        self.hook = {'fc1': [], 'l23_l5': []}
        self.register_hook = False

        self.l23_opto_mask = None
        self.l23_scale = 0
        

    def forward(self, x, td):
        x = torch.cat((x,td), dim=1)
        l23_out = self.activation(self.fc1(x))
        
        if self.l23_opto_mask is not None:
            l23_out = l23_out + torch.ones_like(l23_out)*torch.max(torch.abs(l23_out.detach()))*self.l23_scale*self.l23_opto_mask
        
        l5_pred = self.l23_l5(l23_out)
        
        
        if self.register_hook:
            l23_out.register_hook(lambda grad: self.hook_fn(grad=grad,
                name='fc1'))
            l5_pred.register_hook(lambda grad: self.hook_fn(grad=grad,
                name='l23_l5'))
        
        return l23_out, l5_pred
    
    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': [], 'l23_l5': []}
    
class L5(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, thal_l5_sparsity, ablate_thal_l5, activation, l23_modulation_factor=0.3):
        super().__init__()
        self.latent_dim=latent_dim
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.thal_l5_sparsity = thal_l5_sparsity
        self.ablate_thal_l5 = ablate_thal_l5
        self.activation = get_activation(activation)
        self.hook = {'fc1': []}
        self.register_hook = False
        
        self.l23_modulation_factor=l23_modulation_factor
        
        
    def forward(self, x, l23_pred):
        if self.thal_l5_sparsity == 1. or self.ablate_thal_l5:
            x = x.clone()
            x = self.activation(self.l23_modulation_factor*l23_pred.detach())
            
        elif self.thal_l5_sparsity > 0.:
            bs = x.shape[0]
            x = x.clone().view(-1)
            active_indices = torch.where(abs(x) >= 0.1)[0]
            block = np.random.choice(active_indices.cpu().numpy(), int(active_indices.shape[0]*self.thal_l5_sparsity), replace=False)
            x[block] = 0.
            x = x.view(bs, -1)
            x = self.activation(self.fc1(x) + self.l23_modulation_factor*l23_pred.detach())
        
        else:
            x = self.activation(self.fc1(x) + self.l23_modulation_factor*l23_pred.detach())
            
               
        if self.register_hook:
            x.register_hook(lambda grad: self.hook_fn(grad=grad,
                name='fc1'))
        
        return x
    
    def hook_fn(self, grad, name):
        self.hook[name].append(grad)

    def reset_hook(self):
        self.hook = {'fc1': []}
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,  input_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, input_dim)
    def forward(self, x):
        x = self.fc1(x)
        return x

    
class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, td, activation='sigmoid', thal_l5_sparsity=0., ablate_thal_l5=False, l23_l5_fa=False, fa_sparsity=0., **kwargs):
        super().__init__()
        self.l4 = L4(input_dim, hidden_dim, activation)
        self.l23 = L23(hidden_dim, latent_dim, td, l23_l5_fa, fa_sparsity, activation)
        self.l5 = L5(input_dim,hidden_dim,  latent_dim, thal_l5_sparsity, ablate_thal_l5, activation, l23_modulation_factor=kwargs["l23_modulation_factor"])
        self.decoder = Decoder(latent_dim,hidden_dim, input_dim)
        self.activation = activation
        self.apply(self._init_weights)
        
        
        if fa_sparsity == 1. and l23_l5_fa:
            kwargs['logger'].verbose('freeze l4, l23', 'red')
            for param in self.l4.parameters():
                param.requires_grad = False
            for param in self.l23.parameters():
                param.requires_grad = False
                
                
    def _init_weights(self, m):
        if type(m) == nn.Linear:
            if self.activation == 'sigmoid':
                stdv_weight = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv_weight, stdv_weight)
   
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
                
            elif self.activation == 'relu':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            else:
                raise NotImplementedError
            
    def forward(self, x1, x2, td):
        l4_out = self.l4(x1)
        l23_out, l5_pred = self.l23(l4_out, td)
        l5_out = self.l5(x2 , l5_pred.detach())

        recon = self.decoder(l5_out)
        return l4_out, l23_out, l5_pred, l5_out, recon

    def reset_hook(self):
        self.l23.reset_hook()
        self.l5.reset_hook()
    
    def register_hook(self, register=True):
        self.l23.register_hook = register
        self.l5.register_hook = register


    
    
