import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt

import pathlib
import os

class Digit(Dataset):
    '''
    Sequence of images
    '''
    def __init__(self, digit_mode, n_samples, topdown, noise=0):
        self.n_samples = n_samples - n_samples%10 # divisable by 10 (# digits)
        self.n_digit_samples = n_samples // 10
        assert digit_mode in ['variable','exemplar']
        self.digit_mode = digit_mode
   
        self.mnist_dataset = datasets.MNIST(os.path.join(pathlib.Path(__file__).parent.resolve(),'data'), train=True, download=True)   
        
        self.datasets = []
        self.get_digits()
        self.noise=noise

        assert topdown in ['random','random_walk','none','independent', 'deterministic']
        self.topdown = topdown
        self.topdown_vals = np.arange(0, 10,1)
        self.topdown_input = self.gen_topdown()
        
    def get_digits(self):
        if self.digit_mode == 'exemplar':
            for d in range(10):
                exec("idx_{} = self.mnist_dataset.train_labels=={} ".format(d,d))
                exec("dataset_{} = self.mnist_dataset.train_data[idx_{}]".format(d,d))
                exec("example_digit_{} = dataset_{}[0]".format(d,d))
                exec("dataset_{} = example_digit_{}.view(1,28,28).repeat(self.n_digit_samples,1,1)".format(d,d))
                exec("self.datasets.append(dataset_{})".format(d))
                
        elif self.digit_mode == 'variable':
            for d in range(10):
                exec("idx_{} = self.mnist_dataset.train_labels=={} ".format(d,d))
                exec("dataset_{} = self.mnist_dataset.train_data[idx_{}]".format(d,d))
                exec("dataset_{} = dataset_{}[:self.n_digit_samples]".format(d,d))
                exec("self.datasets.append(dataset_{})".format(d))   

    def gen_topdown(self):
        topdown_vals = self.topdown_vals
        if self.topdown in ['random', 'independent']:
            return np.random.choice(topdown_vals, self.n_samples)
        elif self.topdown == 'random_walk':
            min_td, max_td = min(topdown_vals), max(topdown_vals)
            curr_val = min_td
            vals = [curr_val]
            for _ in range(self.n_samples):
                curr_val += np.random.choice([1,0,-1])
                if curr_val > max_td:
                    curr_val = max_td
                elif curr_val < min_td:
                    curr_val = min_td
                vals.append(curr_val)
            return np.array(vals)

        elif self.topdown == 'deterministic':
            # repeate / tile topdown_vals to match n_samples
            return np.tile(topdown_vals, self.n_samples//len(topdown_vals)+1)[:self.n_samples]

        elif self.topdown=='none':
            return np.zeros(self.n_samples)

    def __getitem__(self, key):
        td1 = self.topdown_input[key]
        td2 = self.topdown_input[key+1]
        if self.topdown in ['random','random_walk', 'deterministic']:
            x1 =  self.datasets[td1][np.random.randint(len(self.datasets[td1]))] / 255.0
            x2 =  self.datasets[td2][np.random.randint(len(self.datasets[td2]))] / 255.0
        else:
            factor1_rnd, factor2_rnd = np.random.choice(self.topdown_vals, 2)
            x1 = self.datasets[factor1_rnd][np.random.randint(len(self.datasets[factor1_rnd]))] / 255.0
            x2 =  self.datasets[factor2_rnd][np.random.randint(len(self.datasets[factor2_rnd]))] / 255.0

        if self.noise > 0:
            x1 = x1 + torch.rand_like(x1)*self.noise
            x2 = x2 + torch.rand_like(x2)*self.noise


        return x1,x2,torch.tensor([td2-td1]), td1, td2


    def __len__(self):
        return self.n_samples-1 # to avoid out of index 


