import numpy as np
import torch.nn.functional as F
import torch
import random
import pickle

from tqdm import tqdm
from pathlib import Path
import os

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def init_model(model, **kwargs):
    return model(**kwargs)


class Logger:
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[98m',
        'end': '\033[0m',
    }
    
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.verbose(f'Experiment: {exp_name}', 'blue')
        
    def verbose(self, msg, color):
        
        if isinstance(msg, dict):
            args = ""
            for k, v in msg.items():
                args += f'{k}: {v}, '
            print(self.COLORS[color] + args + self.COLORS['end'])   
        else:
            print(self.COLORS[color] + msg + self.COLORS['end'])   
        
class Checkpointer:
    def __init__(self,model, dataset, network, exp_name, suffix_args, args, logger, sub_dir=""):
        self.model = model
        self.dataset = dataset
        self.network = network
        self.exp_name = exp_name
        self.suffix_args = suffix_args
        self.args = args
        self.logger = logger
        self.sub_dir = sub_dir
        
        
    def save_cache(self, obj, name):
        save_dir, _, model_suffix = self.get_save_dir()
        cache_dir = os.path.join(Path(save_dir).parent, 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # save obj to cache folder
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
            np.save(os.path.join(cache_dir, f'{name}_{model_suffix}.npy'), obj)
        elif isinstance(obj, np.ndarray):
            np.save(os.path.join(cache_dir, f'{name}_{model_suffix}.npy'), obj)
        elif isinstance(obj, dict):
            with open(os.path.join(cache_dir, f'{name}_{model_suffix}.pkl'), 'wb') as f:
                pickle.dump(obj, f)
        else:
            raise Exception('Unknown type')
        
        
    def get_save_dir(self):

        root_dir = Path(os.path.abspath(__file__)).parent.parent

        base_dir = f'{root_dir}/checkpoints/{self.dataset}/{self.network}/{self.exp_name}'
        model_dir = os.path.join(base_dir, self.sub_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            os.makedirs(os.path.join(model_dir, 'args'))
        if len(self.suffix_args) > 0:
            model_suffix = "_".join([f"{k}_{vars(self.args)[k]}" for k in self.suffix_args])
        else:
            model_suffix = ""
        save_dir = os.path.join(model_dir, f'model_{model_suffix}.pth')
        args_dir = os.path.join(os.path.join(model_dir, 'args'), f'args_{model_suffix}.pkl')
        return save_dir, args_dir, model_suffix
        
    def exists(self):
        return os.path.exists(self.get_save_dir()[0])
    
        
    def save_checkpoint(self):
        save_dir, args_dir, _ = self.get_save_dir()
        self.logger.verbose(f'Saving model to {save_dir}', 'cyan')
        torch.save(self.model.state_dict(), save_dir)
        with open(args_dir, 'wb') as f:
            pickle.dump(self.args, f)
        
    def load_checkpoint(self):
        save_dir, args_dir, _ = self.get_save_dir()
        self.logger.verbose(f'Loading model from {save_dir}', 'cyan')
        self.model.load_state_dict(torch.load(save_dir))
        args = pickle.load(open(args_dir, 'rb'))
        return args
        
    
    
class Trainer:
    def __init__(self, model, dataloader, optimizer, args, logger, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.device = device
        self.ssl_coeff = 1. if not args.ablate_l23_l5 else 0.
        
    def train(self):
        
        self.model.train()
        for e in range(self.args.epoch):
            # store avg loss for each epoch
            ssl_loss_avg = 0
            recon_loss_avg = 0
            for i, (x1,x2,td, _,_) in enumerate(self.dataloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                td = td.to(self.device)
                if self.args.ablate_topdown:
                    td = torch.zeros_like(td)
                    td.to(self.device)
                # flatten x1 and x2
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.view(x2.size(0), -1)
                td = td.view(td.size(0), -1)
                self.optimizer.zero_grad()
                if self.args.ablate_delay:
                    l4_out, l23_out, l5_pred, l5_out, recon = self.model(x2, x2, td)
                else:
                    l4_out, l23_out, l5_pred, l5_out, recon = self.model(x1, x2, td)
                    
                ssl_loss = F.mse_loss(l5_pred, l5_out)
                recon_loss = F.mse_loss(recon, x2)
                
                loss = self.ssl_coeff*ssl_loss + recon_loss
                loss.backward()
                self.optimizer.step()
                ssl_loss_avg += ssl_loss.item()
                recon_loss_avg += recon_loss.item()
                

            ssl_loss_avg /= len(self.dataloader)
            recon_loss_avg /= len(self.dataloader)

        
            if e % 100 == 0:
                print('epoch: {}, ssl_loss: {}, recon_loss: {}'.format(e, ssl_loss_avg, recon_loss_avg))
    
    
    
    
    
class AETrainer:
    def __init__(self, model, dataloader, optimizer, args, logger, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.device = device
        
    def train(self):
        self.model.train()
        for e in range(self.args.epoch):
            # store avg loss for each epoch
            recon_loss_avg = 0
            for i, (_,x2_noisy,_, _,x2_orig) in enumerate(self.dataloader):
                x2_noisy = x2_noisy.to(self.device)
                x2_orig = x2_orig.to(self.device)
                
                # flatten x2
                x2_noisy = x2_noisy.view(x2_noisy.size(0), -1)
                x2_orig = x2_orig.view(x2_orig.size(0), -1)
                self.optimizer.zero_grad()
                recon = self.model(x2_noisy) 
                loss = F.mse_loss(recon, x2_orig)
                
                loss.backward()
                self.optimizer.step()
                recon_loss_avg += loss.item()
                
            recon_loss_avg /= len(self.dataloader)

        
            if e % 100 == 0:
                print('epoch: {}, recon_loss: {}'.format(e, recon_loss_avg))
    