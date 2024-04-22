DOCS = """
train the model on the a dataset (gabor or mnist) and show the robustness of the model to occlusion.
currently figure 4.
"""

from dataset import GaborDatasetOrientation
from model import ShallowMLP
from experiment.utils import set_seed, init_model, Logger, Trainer, Checkpointer
from dataset.utils import OcclusionDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    ## model args
    parser.add_argument('--input_dim', type=int, default=28*28)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--td', type=int, default=1)
    parser.add_argument('--l23_modulation_factor', type=float, default=0.3)
    parser.add_argument('--thal_l5_sparsity', type=float, default=0.0)
    parser.add_argument('--l23_l5_fa',  action='store_true')
    parser.add_argument('--fa_sparsity', type=float, default=0.)
    parser.add_argument('--freeze_thal_l4', action='store_true')
    
    
    # dataset args
    parser.add_argument('--gabor_mode', type=str, default='exemplar')
    parser.add_argument('--topdown_mode', type=str, default='random_walk')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--moving', action='store_true')
    
    
    
    # training args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # experiment args
    parser.add_argument('--ablate_thal_l5', action='store_true')
    parser.add_argument('--ablate_delay', action='store_true')
    parser.add_argument('--ablate_topdown', action='store_true')
    parser.add_argument('--ablate_l23_l5', action='store_true')
    
        
    args = parser.parse_args()
    set_seed(args.seed)
    

    ssl_coeff = 10. if not args.ablate_l23_l5 else 0.
    logger = Logger('experiment for figure 4 / occlusion')
    
    logger.verbose(vars(args), 'magenta')
    
    logger.verbose(f'ablate_thal_l5: {args.ablate_thal_l5}', 'red')
    logger.verbose(f'ablate_delay: {args.ablate_delay}', 'red')
    logger.verbose(f'ablate_topdown: {args.ablate_topdown}', 'red')
    logger.verbose(f'ablate_l23_l5: {args.ablate_l23_l5}', 'red')
    
    
    net = init_model(ShallowMLP,  **vars(args), logger=logger)            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.verbose(f'Using device: {device}', 'yellow')
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    checkpointer = Checkpointer(net, 'gabor', 'shallow_mlp', 'fig4/occlusion', ['seed','ablate_thal_l5','ablate_delay','ablate_topdown','ablate_l23_l5', 'moving'], args, sub_dir="", logger=logger)
    # check if checkpoint exists
    if checkpointer.exists():
        logger.verbose('Checkpoint exists', 'blue')
        if args.continue_training:
            logger.verbose('Continue training from checkpoint', 'blue')
            checkpointer.load_checkpoint()
        else:
            logger.verbose('Do not continue training from checkpoint', 'red')
            sys.exit()
    else:
        logger.verbose('Checkpoint does not exist, start training from scratch', 'blue')

  
    gabor_dataset = GaborDatasetOrientation(gabor_mode=args.gabor_mode, n_samples=2000, topdown=args.topdown_mode, image_size=15, noise=args.noise)
    dataloader_original = DataLoader(gabor_dataset, batch_size=1500, shuffle=True)
    x1_sample, x2_sample, td_sample, x1_label, x2_label = next(iter(dataloader_original))
    # cache the original dataset if data exemplar 
    if args.gabor_mode == 'exemplar':
        logger.verbose('Cache original dataset', 'blue')
        data = {'x1': x1_sample.detach().cpu(), 'x2': x2_sample.detach().cpu(), 'td': td_sample.detach().cpu(),
                'x1_label': x1_label.detach().cpu(), 'x2_label': x2_label.detach().cpu()}
        checkpointer.save_cache(data, 'data')
    dataset = OcclusionDataset(x1_sample.squeeze(), x2_sample.squeeze(), td_sample.squeeze(), x1_label.squeeze(), x2_label.squeeze(),  occlusion_size=0.7, moving=args.moving)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    

    
    trainer = Trainer(net, dataloader, optimizer, args, logger, device)
    trainer.ssl_coeff = 10. if not args.ablate_l23_l5 else 0.
    
    logger.verbose('Start training ...', 'blue')
    trainer.train()
    logger.verbose('Training finished', 'green')
    checkpointer.save_checkpoint()