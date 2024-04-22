from dataset import GaborDatasetOrientation, Digit
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from dataset.utils import NoisyDataset, OcclusionDataset

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from skimage.filters import gabor_kernel
from torchvision import datasets

import pathlib

from experiment.utils import set_seed, init_model, Logger, Checkpointer
from model import ShallowMLP, DenoisingAE

from argparse import Namespace
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import pickle

class CONFIG:
    figsize_single = (2.1, 1.8)
    figsize_double = (2.3, 5)
    axis_fontsize = 8
    legend_fontsize = 7
    title_fontsize = 9
    tick_fontsize = 7

config = CONFIG()



    
def get_labels_gabor():
    thetas = np.linspace(0., np.pi, 10)[:-1]
    labels_imgs = []
    for theta in thetas:
        gk = gabor_kernel(0.2, theta=theta, sigma_x=5, sigma_y=5, offset=0)
        w,h = gk.shape
        mid = (w-1)//2
        gk = gk[mid-6:mid+6,mid-6:mid+6]
        labels_imgs.append(np.real(gk))
    
    return labels_imgs

def get_labels_digit():
    print("directory of dataset to load mnist:", os.path.join(pathlib.Path(__file__).parent.parent.resolve(),'dataset/data'))
    mnist_dataset = datasets.MNIST(os.path.join(pathlib.Path(__file__).parent.parent.resolve(),'dataset/data'), train=True, download=False)   
    
    labels_imgs = []
    for i in range(10):
        labels_imgs.append(-1*mnist_dataset.data[mnist_dataset.targets==i][0].numpy() / 255 - 1)
    return labels_imgs


def load_exp_args(checkpoint_dir):
    args = []
    # open all the .pkl files in checkpoint_dir and add the args as Namespace objects to args
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(checkpoint_dir, file), 'rb') as f:
                args.append(pickle.load(f))
    return args

def plot_task(dataloader, name="task.svg", save_fig=False):
    x1,x2,td,x1_label,x2_label = next(iter(dataloader))
    fig, ax = plt.subplots(2, 32, figsize=(20,5))
    for i in range(32):
            ax[0,i].imshow(x1[i].squeeze().numpy(), cmap='gray')
            ax[0,i].set_title('{}'.format(x1_label[i].item()))
            ax[1,i].imshow(x2[i].squeeze().numpy(), cmap='gray')
            ax[1,i].set_title('{}'.format(x2_label[i].item()))
            # add td
            ax[0,i].text(5.5, 40.5, str(td[i].numpy()), fontsize=10, color='red')
            # remove ticks
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            ax[1,i].set_xticks([])
            ax[1,i].set_yticks([])
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
    
    
def load_dataloader(task_name, args):
    if task_name=="gabor":
        gabor_dataset = GaborDatasetOrientation(gabor_mode=args.gabor_mode, n_samples=2000, topdown=args.topdown_mode, noise=args.noise)
        dataloader = DataLoader(gabor_dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    elif task_name=="mnist":
        mnist_dataset = Digit(digit_mode=args.digit_mode, n_samples=2000, topdown=args.topdown_mode, noise=args.noise)
        dataloader = DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=True)
        return dataloader
    else:
        raise NotImplementedError
    
    
def load_model(model_name, task_name, args, ckpt_dir_dict, device, baseline=False):
    if model_name == "shallow_mlp":
        logger = Logger('dummy logger for loading model')
        model = init_model(ShallowMLP,  **vars(args), logger=logger)
        if not baseline:
            checkpointer = Checkpointer(model, task_name, model_name, ckpt_dir_dict['exp_name'], ckpt_dir_dict['suffix_args'], args, logger, ckpt_dir_dict['sub_dir'])
            checkpointer.load_checkpoint()
        else:
            print("\nNOT LOADING CHECKPOINT => RANDOM MODEL\n")
        return model.to(device)
    
def run_eval(model_name, task_name, args, ckpt_dir_dict, device, no_input=False, no_topdown=False, baseline=False):
    set_seed(args.seed)
    
    net = load_model(model_name, task_name, args, ckpt_dir_dict, device, baseline=baseline)
    net.eval()


    dataloader = load_dataloader(task_name, args)

    l5_out_all = []
    l23_out_all = []
    l4_out_all = []
    x1_label_all = []
    x2_label_all = []
    td_all = []


    for i, (x1,x2,td,  x1_label, x2_label) in enumerate(dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        td = td.to(device)
        # flatten x1 and x2
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        td = td.view(td.size(0), -1)
        
        if no_input:
            x1 = torch.randn_like(x1) * 0.01
            x2 = torch.randn_like(x2) * 0.01
        if no_topdown:
            td = torch.randn_like(td.float()) * 0.01
            
        if args.ablate_topdown:
                td = torch.zeros_like(td)
                td.to(device)
                
        if args.ablate_delay:
            l4_out, l23_out, l23_l5_out, l5_out, recon = net(x2, x2, td)
        else:
            l4_out, l23_out, l23_l5_out, l5_out, recon = net(x1, x2, td)
            
        l5_out_all.append(l5_out.detach().cpu().numpy())
        l23_out_all.append(l23_out.detach().cpu().numpy())
        l4_out_all.append(l4_out.detach().cpu().numpy())
        x2_label_all.append(x2_label.detach().cpu().numpy())
        x1_label_all.append(x1_label.detach().cpu().numpy())
        td_all.append(td.squeeze().detach().cpu().numpy())

    l5_out_all = np.concatenate(l5_out_all, axis=0)
    x2_label_all = np.concatenate(x2_label_all, axis=0)
    x1_label_all = np.concatenate(x1_label_all, axis=0)
    l23_out_all = np.concatenate(l23_out_all, axis=0)
    l4_out_all = np.concatenate(l4_out_all, axis=0)
    td_all = np.concatenate(td_all, axis=0)

    return l5_out_all, l23_out_all, l4_out_all, x1_label_all, x2_label_all, td_all


def get_pca_variances(exp_args, model_name, task_name, ckpt_dir_dict):
    args_cols = [k for k in vars(exp_args[0]).keys()]
    df_res = []
    for args in exp_args:
        l5_out_all, l23_out_all, _, x1_label_all, x2_label_all, td_all = run_eval(model_name=model_name, task_name=task_name, args=args,ckpt_dir_dict=ckpt_dir_dict, device=torch.device('cpu'))
        
        clf_l23 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l23_out_all, x2_label_all)
        clf_l5 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l5_out_all, x2_label_all)

        # accumatulate variance explained
        pca_l23 = PCA(n_components=args.hidden_dim)
        pca_l23.fit(l23_out_all)
        pca_l5 = PCA(n_components=args.latent_dim)
        pca_l5.fit(l5_out_all)

        df_res.append(
            [clf_l23.score(l23_out_all, x2_label_all), clf_l5.score(l5_out_all, x2_label_all), 
                                pca_l23.explained_variance_ratio_, 
                                pca_l5.explained_variance_ratio_]+[ v for k, v in vars(args).items()])
    
    df = pd.DataFrame(df_res, columns=['l23_acc', 'l5_acc', 'l23_accum_var_explained', 'l5_accum_var_explained']+args_cols)
    return df




def get_noise_metric(exp_args):
    args_cols = [k for k in vars(exp_args[0]).keys()]
    df_res = []
    recons_noise = {True:{}, False:{}}
    for args in exp_args:
        recons_noise[args.ablate_l23_l5][args.added_noise] = {}
        logger = Logger('dummy logger for loading model')
        set_seed(args.seed)
        
        net = init_model(ShallowMLP,  **vars(args), logger=logger)         

        device = torch.device("cpu")
        logger.verbose(f'Using device: {device}', 'yellow')
        
        checkpointer = Checkpointer(net, 'gabor', 'shallow_mlp', 'fig4/noise', ['seed','ablate_thal_l5','ablate_delay','ablate_topdown','ablate_l23_l5', 'added_noise'], args, sub_dir="", logger=logger)
        checkpointer.load_checkpoint()
        net.to(device)
        
        
        save_dir, _, model_suffix = checkpointer.get_save_dir()
        cache_dir = os.path.join(Path(save_dir).parent, 'cache')
        
        with open(os.path.join(cache_dir, f'data_{model_suffix}.pkl'), 'rb') as f:
            data_samples = pickle.load(f)
            x1_sample = data_samples['x1']
            x2_sample = data_samples['x2']
            td_sample = data_samples['td']

        dataset = NoisyDataset(x1_sample.squeeze(), x2_sample.squeeze(), td_sample.squeeze() ,noise=args.added_noise)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        # get average of resid_loss for all the samples after the training
        with torch.no_grad():
            resid_loss_l5 = []
            resid_loss_l23 = []

            l23_recons = []
            l5_recons = []

            x1_origs = []
            x2_origs = []
            x1_noisy = []
            x2_noisy = []
            
            
            for data in dataloader:
                x1, x2, td, x1_orig, x2_orig = data
                x1_noisy.append(x1.numpy())
                x2_noisy.append(x2.numpy())
                x1_origs.append(x1_orig.numpy())
                x2_origs.append(x2_orig.numpy())
                x1, x2, td = x1.view(-1,784), x2.view(-1,784), td.view(-1,1)
                x1, x2,td = x1.to(device), x2.to(device), td.to(device)
                if args.ablate_topdown:
                    td = torch.zeros_like(td)
                if args.ablate_delay:

                    l4_out, l23_out, l23_l5_out, l5_out, recon = net(x2, x2, td)
                else:
                    l4_out, l23_out, l23_l5_out, l5_out, recon = net(x1, x2, td)
                
                if args.ablate_thal_l5:
                    l4_out, l23_out, l23_l5_out, l5_out, recon = net(x1, torch.rand_like(x2), td)
                    
                l23_recon = net.decoder(l23_l5_out)
                resid_loss_l5.append(np.linalg.norm(recon.detach().cpu().numpy() - x2_orig.view(-1, 784).detach().cpu().numpy(), axis=1))
                resid_loss_l23.append(np.linalg.norm(l23_recon.detach().cpu().numpy() - x2_orig.view(-1, 784).detach().cpu().numpy(), axis=1))
                l23_recons.append(l23_recon.view(-1, 28, 28).detach().cpu().numpy())
                l5_recons.append(recon.view(-1, 28, 28).detach().cpu().numpy())
            resid_loss_l5 = np.concatenate(resid_loss_l5)
            resid_loss_l23 = np.concatenate(resid_loss_l23)
            print('Average Residual Loss | L5: : {:.4f}'.format(resid_loss_l5.mean()))
            print('Average Residual Loss | L23: : {:.4f}'.format(resid_loss_l23.mean()))
            
            resid_loss_l5 = resid_loss_l5.mean()
            resid_loss_l23 = resid_loss_l23.mean()
        

        df_res.append([resid_loss_l23, resid_loss_l5]+[ v for k, v in vars(args).items()])
        recons_noise[args.ablate_l23_l5][args.added_noise] = {'l23_recons': np.concatenate(l23_recons), 'l5_recons': np.concatenate(l5_recons), 'x1_orig': np.concatenate(x1_origs), 'x2_orig': np.concatenate(x2_origs), 'x1_noisy': np.concatenate(x1_noisy), 'x2_noisy': np.concatenate(x2_noisy)}
    
    df = pd.DataFrame(df_res, columns=['l23_residual', 'l5_residual']+args_cols)
    return df, recons_noise



def get_noise_metric_ae(exp_args):
    args_cols = [k for k in vars(exp_args[0]).keys()]
    df_res = []
    recons_noise = {}
    for args in exp_args:
        recons_noise[args.added_noise] = {}
        logger = Logger('dummy logger for loading model')
        set_seed(args.seed)
        
        net = init_model(DenoisingAE,  **vars(args))         

        device = torch.device("cpu")
        logger.verbose(f'Using device: {device}', 'yellow')
        
        checkpointer = Checkpointer(net, 'gabor', 'shallow_mlp', 'fig4/noise', ['seed','ablate_thal_l5','ablate_delay','ablate_topdown','ablate_l23_l5', 'added_noise'], args, sub_dir="ae", logger=logger)
        checkpointer.load_checkpoint()
        net.to(device)
        
        
        save_dir, _, model_suffix = checkpointer.get_save_dir()
        cache_dir = os.path.join(Path(save_dir).parent, 'cache')
        
        with open(os.path.join(cache_dir, f'data_{model_suffix}.pkl'), 'rb') as f:
            data_samples = pickle.load(f)
            x1_sample = data_samples['x1']
            x2_sample = data_samples['x2']
            td_sample = data_samples['td']

        dataset = NoisyDataset(x1_sample.squeeze(), x2_sample.squeeze(), td_sample.squeeze() ,noise=args.added_noise)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        # get average of resid_loss for all the samples after the training
        with torch.no_grad():
            resid_loss = []
            recons = []
            x2_origs = []
            x2_noisy = []
            
            
            for data in dataloader:
                _, x2, _, _, x2_orig = data
                x2_noisy.append(x2.numpy())
                x2_origs.append(x2_orig.numpy())
                x2 = x2.view(-1,784)
                x2 = x2.to(device)
                recon = net(x2)
                resid_loss.append(np.linalg.norm(recon.detach().cpu().numpy() - x2_orig.view(-1, 784).detach().cpu().numpy(), axis=1))
                recons.append(recon.view(-1, 28, 28).detach().cpu().numpy())
 
            resid_loss = np.concatenate(resid_loss)
            print('Average Residual Loss : {:.4f}'.format(resid_loss.mean()))
            
            resid_loss = resid_loss.mean()
        

        df_res.append([resid_loss]+[ v for k, v in vars(args).items()])
        recons_noise[args.added_noise] = {'ae_recons': np.concatenate(recons), 'x2_orig': np.concatenate(x2_origs), 'x2_noisy': np.concatenate(x2_noisy)}
    
    df = pd.DataFrame(df_res, columns=['ae_residual']+args_cols)
    return df, recons_noise





def get_occlusion_metric(exp_args):
    args_cols = [k for k in vars(exp_args[0]).keys()]
    df_res = []
    recons_occlusion = {"ablate_l23_l5": {True:{}, False:{}}, "ablate_delay": {True:{}, False:{}}}
    for args in exp_args:
        if "moving" not in args: # hacky way to add moving to the args for moving=True
            args.moving = True
        recons_occlusion["ablate_l23_l5"][args.ablate_l23_l5] = {}
        recons_occlusion["ablate_delay"][args.ablate_delay] = {}
        
        logger = Logger('dummy logger for loading model')
        set_seed(args.seed)
        
        net = init_model(ShallowMLP,  **vars(args), logger=logger)            

        device = torch.device("cpu")
        logger.verbose(f'Using device: {device}', 'yellow')
        
        checkpointer = Checkpointer(net, 'gabor', 'shallow_mlp', 'fig4/occlusion', ['seed','ablate_thal_l5','ablate_delay','ablate_topdown','ablate_l23_l5', 'moving'], args, sub_dir="", logger=logger)
        checkpointer.load_checkpoint()
        net.to(device)
        
        
        save_dir, _, model_suffix = checkpointer.get_save_dir()
        cache_dir = os.path.join(Path(save_dir).parent, 'cache')
        if args.gabor_mode == 'exemplar':
            logger.verbose('Load cached dataset', 'blue')
            with open(os.path.join(cache_dir, f'data_{model_suffix}.pkl'), 'rb') as f:
                data_samples = pickle.load(f)
                x1_sample = data_samples['x1']
                x2_sample = data_samples['x2']
                td_sample = data_samples['td']
                x1_label = data_samples['x1_label']
                x2_label = data_samples['x2_label']
        else:
            gabor_dataset = GaborDatasetOrientation(gabor_mode=args.gabor_mode, n_samples=1100, topdown=args.topdown_mode, image_size=15, base_freq=0.2, noise=args.noise)
            dataloader_original = DataLoader(gabor_dataset, batch_size=1900, shuffle=True)
            x1_sample, x2_sample, td_sample, x1_label, x2_label = next(iter(dataloader_original))

        dataset = OcclusionDataset(x1_sample.squeeze(), x2_sample.squeeze(), td_sample.squeeze(), x1_label.squeeze(), x2_label.squeeze(), occlusion_size=0.7, moving=args.moving)
        dataloader = DataLoader(dataset, batch_size=x1_sample.shape[0], shuffle=True, drop_last=True)

        # get average of resid_loss for all the samples after the training
        with torch.no_grad():

            x1, x2, td, x1_label, x2_label = next(iter(dataloader))
            x1_test, x2_test, td_test, x1_label_test, x2_label_test = next(iter(dataloader))
            x1, x2 , td = x1.view(-1,784).to(device), x2.view(-1,784).to(device), td.view(-1,1).to(device)
            x1_test , x2_test , td_test = x1_test.view(-1,784).to(device), x2_test.view(-1,784).to(device), td_test.view(-1,1).to(device)

            if args.ablate_delay:
                l4_out_train, l23_out_train, l23_l5_out_train, l5_out_train, recon_train = net(x2, x2, td)
            
            else:
                l4_out_train, l23_out_train, l23_l5_out_train, l5_out_train, recon_train = net(x1, x2, td)
            l23_recon_train = net.decoder(l23_l5_out_train)
            
            if args.ablate_delay:
                l4_out_test, l23_out_test, l23_l5_out_test, l5_out_test, recon_test = net(x2_test, x2_test, td_test)
            else:
                l4_out_test, l23_out_test, l23_l5_out_test, l5_out_test, recon_test = net(x1_test, x2_test, td_test)
                
            l23_recon_test = net.decoder(l23_l5_out_test)
            
            l23_recons=l23_recon_train.view(-1, 28, 28).detach().cpu().numpy()
            l5_recons=recon_train.view(-1, 28, 28).detach().cpu().numpy()
            # create linear regression object
            logger.verbose('Train SVM L23', 'yellow')
            clf_l23_x2 = svm.SVC(kernel='poly', degree=6, gamma='scale', max_iter=10000).fit(l23_l5_out_train.detach().cpu().numpy(), x2_label.detach().cpu().numpy())
            logger.verbose('Train SVM L5', 'yellow')
            clf_l5_x2 = svm.SVC(kernel='poly', degree=6, gamma='scale', max_iter=10000).fit(l5_out_train.detach().cpu().numpy(), x2_label.detach().cpu().numpy())
            logger.verbose('Train SVM L23 X1', 'yellow')
            
            clf_l23_x1 = svm.SVC(kernel='poly', degree=6, gamma='scale', max_iter=10000).fit(l23_l5_out_train.detach().cpu().numpy(), x1_label.detach().cpu().numpy())
            logger.verbose('Train SVM L5 X1', 'yellow')
            
            clf_l5_x1 = svm.SVC(kernel='poly', degree=6, gamma='scale', max_iter=10000).fit(l5_out_train.detach().cpu().numpy(), x1_label.detach().cpu().numpy())
            
            
            l23_accuracy_x2 = accuracy_score(x2_label_test.detach().cpu().numpy(), clf_l23_x2.predict(l23_l5_out_test.detach().cpu().numpy()).round())
            l5_accuracy_x2 = accuracy_score(x2_label_test.detach().cpu().numpy(), clf_l5_x2.predict(l5_out_test.detach().cpu().numpy()).round())
            
            l23_accuracy_x1 = accuracy_score(x1_label_test.detach().cpu().numpy(), clf_l23_x1.predict(l23_l5_out_test.detach().cpu().numpy()).round())
            l5_accuracy_x1 = accuracy_score(x1_label_test.detach().cpu().numpy(), clf_l5_x1.predict(l5_out_test.detach().cpu().numpy()).round())
        

        df_res.append([l23_accuracy_x2,  l5_accuracy_x2, l23_accuracy_x1, l5_accuracy_x1]+[ v for k, v in vars(args).items()])
        if not args.ablate_delay:
            recons_occlusion["ablate_l23_l5"][args.ablate_l23_l5] = {'l23_recons': l23_recons, 'l5_recons': l5_recons, 'x1_noisy': x1.numpy(), 'x2_noisy': x2.numpy()}
        if not args.ablate_l23_l5:
            recons_occlusion["ablate_delay"][args.ablate_delay] = {'l23_recons': l23_recons, 'l5_recons': l5_recons, 'x1_noisy': x1.numpy(), 'x2_noisy': x2.numpy()}
        
    
    df = pd.DataFrame(df_res, columns=['l23_accuracy_x2', 'l5_accuracy_x2','l23_accuracy_x1', 'l5_accuracy_x1']+args_cols)
    return df, recons_occlusion




def get_activations(exp_args, model_name, task_name, ckpt_dir_dict, hidden_dim=None, dynamic_ablation=False, baseline=False):
    activations = {}

    conditions = [[False, False]]
    if dynamic_ablation:
        print("Dynamic ablation is ON...")
        conditions = [[False, False], [True, False], [False, True], [True, True]]
        
    for no_input, no_topdown in conditions:
        
        k = f"{no_input}-{no_topdown}"
        activations[k] = {}
        for args in exp_args:
            if hidden_dim is not None:
                if (args.hidden_dim not in  hidden_dim) or (args.latent_dim != args.hidden_dim):
                    continue
                print(f"{args.hidden_dim=} / {args.latent_dim=}")
            if args.seed not in activations[k]:
                activations[k][args.seed] = {}
            # args.batch_size=1
            l5_out_all, l23_out_all, l4_out_all, x1_label_all, x2_label_all, td_all = run_eval(model_name=model_name, task_name=task_name, args=args,ckpt_dir_dict=ckpt_dir_dict, device=torch.device('cpu'), no_input=no_input, no_topdown=no_topdown,baseline=baseline)
            
            activations[k][args.seed][args.hidden_dim] = {"l23_activations": l23_out_all, "l5_activations": l5_out_all, "l4_activations": l4_out_all}

    return activations



def get_prediction(exp_args, model_name, task_name, ckpt_dir_dict):
    df_all = pd.DataFrame()
    if task_name=="gabor":
        n_classes = 9
        conf_matrix_avg = np.zeros((9,9))
    elif task_name=="mnist":
        n_classes = 10
        conf_matrix_avg = np.zeros((10,10))
    else:
        raise NotImplementedError
    
    
    for args in exp_args:
        l5_out_all, l23_out_all, l4_out_all, x1_label_all, x2_label_all, td_all = run_eval(model_name=model_name, task_name=task_name, args=args,ckpt_dir_dict=ckpt_dir_dict, device=torch.device('cpu'))
        
        
        ## preds l23 and l5 for x2
        clf_l23 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l23_out_all, x2_label_all)
        clf_l5 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l5_out_all, x2_label_all)


        clf_l23.score(l23_out_all, x2_label_all)
        preds_l23 = clf_l23.predict(l23_out_all)
        clf_l5.score(l5_out_all, x2_label_all)
        preds_l5 = clf_l5.predict(l5_out_all)
        
        
        ## preds l23 and l5 for x1
        clf_l23_x1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l23_out_all, x1_label_all)
        clf_l5_x1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(l5_out_all, x1_label_all)
        
        clf_l23_x1.score(l23_out_all, x1_label_all)
        preds_l23_x1 = clf_l23_x1.predict(l23_out_all)
        clf_l5_x1.score(l5_out_all, x1_label_all)
        preds_l5_x1 = clf_l5_x1.predict(l5_out_all)
        
        

        data = {"preds_l23": preds_l23,"preds_l5": preds_l5, 
                'preds_l23_x1': preds_l23_x1, 'preds_l5_x1': preds_l5_x1 ,
                "td": td_all, "x1_label": x1_label_all, "x2_label": x2_label_all,
                "l23_sparsity": TR_measure(l23_out_all),
                "l4_sparsity": TR_measure(l4_out_all),
                "l5_sparsity": TR_measure(l5_out_all)}
        # create a DataFrame
        df = pd.DataFrame({
            'x1': data['x1_label'],
            'x2': data['x2_label'],
            'cue': data['td'],
            'preds_l23': data['preds_l23'],
            'preds_l5': data['preds_l5'],
            'preds_l23_x1': data['preds_l23_x1'],
            'preds_l5_x1': data['preds_l5_x1'],
            'l23_sparsity': data['l23_sparsity'],
            'l4_sparsity': data['l4_sparsity'],
            'l5_sparsity': data['l5_sparsity'],
            **{k: [v] * len(data['x1_label']) for k, v in vars(args).items()}
        })
        df_all = pd.concat([df_all, df], axis=0)
        
        conf_matrix = confusion_matrix(x2_label_all, preds_l5, labels=np.arange(n_classes))
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_avg += conf_matrix
    conf_matrix_avg /= len(exp_args)
    
    return df_all, conf_matrix_avg


def plot_l23_prediction(data_df, name, task, plot_kwargs={"x_box_alignment":(0.5, 2.5), "y_box_alignment": (2.3, 0.5), "figsize":(2.3, 2.3)}, save_fig=False):
    predictions = data_df.copy()
    num_seeds = len(predictions['seed'].unique())
    
    if task == 'gabor':
        label_images = get_labels_gabor()
    elif task == 'mnist':
        label_images = get_labels_digit()
    else:
        raise NotImplementedError
    
    predictions['mean_l23'] = predictions.groupby(['x1', 'x2', 'cue'])['preds_l23'].transform('mean')
    predictions['std_l23'] = predictions.groupby(['x1', 'x2', 'cue'])['preds_l23'].transform('std')
    predictions['mean_l5'] = predictions.groupby(['x1', 'x2', 'cue'])['preds_l5'].transform('mean')
    predictions['std_l5'] = predictions.groupby(['x1', 'x2', 'cue'])['preds_l5'].transform('std')
    predictions = predictions.drop(['preds_l23','preds_l5', 'seed'], axis=1)
    predictions = predictions.drop_duplicates()
    predictions

    plt.figure(figsize=plot_kwargs['figsize'])
    colors = ['#5fa6d1ff', 'gray','#e9724cff' ]
    # Create a scatter plot and fit a line for each cue
    for cue, color in zip([-1, 0, 1], colors):
        sns.regplot(x=predictions[predictions['cue']==cue]['x1'], y=predictions[predictions['cue']==cue]['mean_l23'], scatter_kws={'color': color, 's':10}, line_kws={'color': color}, label=f'cue={cue}')
    # add error bars for each cue, with the same color as the line
    for cue, color in zip([-1, 0, 1], colors):
        plt.errorbar(predictions[predictions['cue']==cue]['x1'], predictions[predictions['cue']==cue]['mean_l23'], yerr=predictions[predictions['cue']==cue]['std_l23']/np.sqrt(num_seeds), fmt='.', color=color)
    # Set the title and labels
    # plt.title('Predictions for different cue values')
    plt.xlabel('$\mathrm{input}\: (x_{t-1})$', fontsize=config.axis_fontsize, labelpad=10)
    plt.ylabel('$\mathrm{predicted\, input}\:(\hat{x}_t)$', fontsize=config.axis_fontsize, labelpad=10)

    # plt.xticks(range(0, 9), fontsize=config.tick_fontsize)
    # plt.yticks(range(0, 9), fontsize=config.tick_fontsize)
    plt.xticks(range(0, len(label_images)), [])
    plt.yticks(range(0, len(label_images)), [])
    
    ax = plt.gca()
    tick_labels = ax.xaxis.get_ticklabels()

    for i,im in enumerate(label_images):
        ib = OffsetImage(im, zoom=.7, cmap='gray')
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment= plot_kwargs['x_box_alignment'],
                        # pad=0.01,
                        )
        ax.add_artist(ab)
    
    # same for y-axis
    tick_labels = ax.yaxis.get_ticklabels()
    for i,im in enumerate(label_images):
        ib = OffsetImage(im, zoom=.7, cmap='gray')
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=plot_kwargs['y_box_alignment']
                        )
        ax.add_artist(ab)
    

    plt.grid(False)
    # remove top and right spines
    sns.despine()
    
    
    
     # colors = ['#5fa6d1ff', 'gray','#e9724cff' ]
    orange_circle = mlines.Line2D([], [], color='#e9724cff', marker='o', linestyle='None',
                            markersize=4, label=r'$+18^{\circ}$')
    gray_circle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                            markersize=4, label=r'$0^{\circ}$')
    blue_circle = mlines.Line2D([], [], color='#5fa6d1ff', marker='o', linestyle='None',
                            markersize=4, label=r'$-18^{\circ}$')

    marker_legend =plt.legend(handles=[orange_circle, gray_circle, blue_circle], title='top-down', fontsize=config.legend_fontsize, title_fontsize=config.legend_fontsize)
    # add both legends back
    plt.gca().add_artist(marker_legend)
    plt.tight_layout()


    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
        
def plot_l5_prediction(conf_matrix_avg, name, task, plot_kwargs={"x_box_alignment":(0.5, 1.8), "y_box_alignment": (1.8, 0.5), "figsize": (2.2,1.9)}, save_fig=False):
    n_classes = conf_matrix_avg.shape[0]
    df_cm = pd.DataFrame(conf_matrix_avg, index = [str(i) for i in np.arange(n_classes)],
                    columns = [str(i) for i in np.arange(n_classes)])
    
    if task == 'gabor':
        label_images = get_labels_gabor()
    elif task == 'mnist':
        label_images = get_labels_digit()
    else:
        raise NotImplementedError
    
    plt.figure(figsize = plot_kwargs['figsize'])
    sns.heatmap(df_cm, annot=False, fmt=".3g", cbar_kws={'format': '%.0f'}, vmin=0, vmax=100, cmap='Blues', linewidths=0.01, linecolor='gray', clip_on=False)
    # cbar fontsize
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=config.tick_fontsize)
    # add text to cbar
    cbar.set_title('acc.(%)', fontsize=config.axis_fontsize)
    # swap yaxis
    plt.gca().invert_yaxis()
    plt.xlabel('$\mathrm{input}\: (x_t)$', fontsize=config.axis_fontsize, labelpad=10)
    plt.ylabel('$\mathrm{predicted\, input}\:(\hat{x}_t)$', fontsize=config.axis_fontsize, labelpad=10)

    plt.xticks(np.arange(0.5, len(label_images)+.5, 1), [])
    plt.yticks(np.arange(0.5, len(label_images)+.5, 1),[])

    ax = plt.gca()
    tick_labels = ax.xaxis.get_ticklabels()

    for i,im in enumerate(label_images):
        ib = OffsetImage(im, zoom=.7, cmap='gray')
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment= plot_kwargs['x_box_alignment'],
                        # pad=0.01,
                        )
        ax.add_artist(ab)
    
    # same for y-axis
    tick_labels = ax.yaxis.get_ticklabels()
    for i,im in enumerate(label_images):
        ib = OffsetImage(im, zoom=.7, cmap='gray')
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=plot_kwargs['y_box_alignment']
                        )
        ax.add_artist(ab)

    ax = plt.gca()
    # plt.title("Confusion matrix for each digit by L5 output")
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
        
        
        
def plot_residual(results, results_ae, x_axis, y_axis, x_label, name, hue="ablate_l23_l5", save_fig=False):
    plt.figure(figsize=(2, 2))
    # palette = sns.color_palette("mako_r", 6)
    ax = sns.lineplot(x=x_axis, y=y_axis, hue=hue, data=results, errorbar='se',  palette=['#1e1f1c', '#bc5090'], linewidth=1.5, marker='o', markersize=5, legend=True)
    # add another line for the AE
    sns.lineplot(x=x_axis, y='ae_residual', data=results_ae, color='black', linewidth=1., marker='o', markersize=5, legend=True,linestyle='--')
    ax.plot(markersize=1)
    ax.set_xlabel(x_label, fontsize= config.axis_fontsize)
    ax.set_ylabel('reconstruction residual', fontsize= config.axis_fontsize)

    # set ticks fontsize
    plt.xticks(fontsize=8)
    # set range of y-axis
    # plt.ylim(0, 6)
    plt.yticks(fontsize=8)

    sns.despine()
    # add legend #1e1f1c = "full" #bc5090 = "rL2/3 $\rightarrow$ L5 k.o."
    custom_lines = [mlines.Line2D([0], [0], color="#1e1f1c"),
                mlines.Line2D([0], [0], color="#bc5090"),
                mlines.Line2D([0], [0], color="black", linestyle='--')]
    # full_patch = mpatches.Patch(facecolor='white', label='full', edgecolor='#1e1f1c')
    # ablation_patch = mpatches.Patch(facecolor='white', label=r'L2/3 $\rightarrow$ L5 k.o.', edgecolor='#bc5090')
    if hue=="ablate_l23_l5":
        legend_label = r'L2/3 $\rightarrow$ L5 k.o.' 
    
    elif hue=="ablate_delay":
        legend_label = r'delay k.o.' 
    else:
        raise NotImplemented
    ax.legend(custom_lines, ['full',legend_label, 'denoising AE'] ,fontsize=config.legend_fontsize, title_fontsize=config.legend_fontsize)
    

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        

def plot_occlusion_accuracy_plot(results, name, hue="ablate_l23_l5", save_fig=False):
    df = results
    # Melt the DataFrame to long format for easier plotting
    melted_df = pd.melt(df, id_vars=['seed', hue], var_name='layer_accuracy', value_name='accuracy')
    filtered_df = melted_df[melted_df['layer_accuracy'].isin(['l23_accuracy_x2', 'l5_accuracy_x2'])]
    filtered_df['layer_accuracy'].replace({'l23_accuracy_x2': 'L2/3', 'l5_accuracy_x2': 'L5'}, inplace=True)
    # convert to percentage
    filtered_df['accuracy'] *= 100
    
    # Create a bar plot
    plt.figure(figsize=(1.7, 1.7))
    ax = sns.barplot(x='layer_accuracy', y='accuracy', hue=hue, data=filtered_df)

    # Define the colors and hatches manually - replace these with your preferred colors and hatches
    palette = ['#1e1f1c', '#1e1f1c', '#bc5090', '#bc5090']  # replace these colors as needed
    hatches = ['', '', '/', '/']  # replace these hatches as needed 

    # Iterate over bars and manually change color and hatches
    for i, bar in enumerate(ax.patches):
        # set colors
        bar.set_facecolor(palette[i % len(palette)])
        bar.set_hatch(hatches[i % len(hatches)])
        bar.set_edgecolor('white')

    # Set plot labels and title
    ax.set_ylabel(r'$x_t$ decoding acc. (%)', fontsize=config.axis_fontsize)
    ax.set_xlabel('', fontsize=config.axis_fontsize)
    # set values [0, 20, 40, 60, 80, 100] to y-axix
    ax.set_yticks(np.arange(0, 101, 20))
    # set fontsize of y-axis ticks
    ax.tick_params(axis='y', labelsize=config.tick_fontsize)
    # set fontsize of x-axis ticks
    ax.tick_params(axis='x', labelsize=config.tick_fontsize)
    
    if hue=="ablate_l23_l5":
        legend_label = r'L2/3 $\rightarrow$ L5 k.o.' 
    
    elif hue=="ablate_delay":
        legend_label = r'delay k.o.' 
    else:
        raise NotImplemented
    ablation_patch = mpatches.Patch(facecolor='#bc5090', hatch="//", label=legend_label, edgecolor='white')
    full_patch = mpatches.Patch(facecolor='#1e1f1c', label='full', edgecolor='black')
    ax.legend(handles=[ablation_patch, full_patch], fontsize=config.legend_fontsize, title_fontsize=config.legend_fontsize)
    
    # remvoe top and right spines
    sns.despine()
    
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
        
        
def plot_accuracy_per_connection_probability(results, name, save_fig=False):
    df_acc = results[['l23_acc','l5_acc', 'seed', 'RND', 'Sparsity']]
    df_acc['Layer'] = ['L2/3']*len(df_acc)
    df_acc['l23_acc'] = df_acc['l23_acc'].astype(float) * 100
    df_acc['l5_acc'] = df_acc['l5_acc'].astype(float) * 100
    
    df_acc['Connection_prob'] = 1 - df_acc['Sparsity'].astype(float)

    df_acc['RND'] = df_acc['RND'].astype(bool)
    # add 0.1 for the connectin_prob where the RND column if False
    df_acc.loc[df_acc['RND'] == False, 'Connection_prob'] = 'optimal'

    df_acc_fa = df_acc[df_acc['RND']]
    df_acc_bp = df_acc[~df_acc['RND']]

    # plot barplot as well
    fig, ax = plt.subplots(2,1, figsize=(2.5, 4))
    colors = ["#b3b3b3", "#b3b3b3"]
    # sns.barplot(data=df_acc, x='Connection_prob', y='acc', hue='Layer', errorbar='se', errwidth=1.2, width=0.5, palette=sns.color_palette(colors))
    sns.lineplot(data=df_acc_fa, x='Connection_prob', y='l23_acc',  markers=True, dashes=False,err_style="bars", errorbar="se", color='#b3b3b3', label=r'random L5$\rightarrow$L2/3 feedback', ax=ax[0])
    sns.lineplot(data=df_acc_fa, x='Connection_prob',  y='l5_acc',  markers=True, dashes=False,err_style="bars", errorbar="se", color='#b3b3b3', label=r'random L5$\rightarrow$L2/3 feedback', ax=ax[1])
    # add horizontal line for optimal
    ax[0].axhline(y=np.mean(df_acc_bp['l23_acc']), color='steelblue', linestyle='--', label='optimal')
    ax[1].axhline(y=np.mean(df_acc_bp['l5_acc']), color='steelblue', linestyle='--', label='optimal')
    # remove spines
    ax[0].set_xlabel('$ W_{L5 \\rightarrow L2/3}^{RND} \, \mathrm{connection}\,\mathrm{probability}$', fontsize=config.axis_fontsize)
    ax[0].set_ylabel(r'$x_2$ decoding acc.(%)',fontsize=config.axis_fontsize)
    ax[1].set_xlabel('$ W_{L5 \\rightarrow L2/3}^{RND} \, \mathrm{connection}\,\mathrm{probability}$', fontsize=config.axis_fontsize)
    ax[1].set_ylabel(r'$x_2$ decoding acc.(%)',fontsize=config.axis_fontsize)
    # set ticks fontsize
    ax[0].tick_params(axis='y', labelsize=config.tick_fontsize)
    ax[0].tick_params(axis='x', labelsize=config.tick_fontsize)
    ax[1].tick_params(axis='y', labelsize=config.tick_fontsize)
    ax[1].tick_params(axis='x', labelsize=config.tick_fontsize)
    
    # set range of values on y-axis to be 0-20-40-60-80-100
    ax[0].set_yticks([0, 50, 100])
    ax[1].set_yticks([0, 50, 100])
    ax[0].set_xticks([0, 0.5, 1])
    ax[1].set_xticks([0, 0.5, 1])
    ax[0].set_xlim(0, 1.)
    ax[1].set_xlim(0, 1.)
    

    ax[0].set_title('L2/3', fontsize=config.title_fontsize)
    ax[1].set_title('L5', fontsize=config.title_fontsize)
    #change legend fontsize
    ax[0].legend(fontsize=config.legend_fontsize)
    ax[1].legend(fontsize=config.legend_fontsize)
    
    sns.despine()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
def plot_explained_variance_connection_probablity(results, name, save_fig):
    
    df_pca = results[['l23_accum_var_explained','l5_accum_var_explained', 'seed', 'RND', 'Sparsity']].rename(columns={"l23_accum_var_explained": 'l23_exp_var', "l5_accum_var_explained": 'l5_exp_var'})
    df_pca['Connection_prob'] = 1 - df_pca['Sparsity'].astype(float)
    df_pca['RND'] = df_pca['RND'].astype(bool)
    # filter out connection_prob in [0.25, 0.5, 0.75]
    df_pca = df_pca[df_pca['Connection_prob'] != 0.25]
    df_pca = df_pca[df_pca['Connection_prob'] != 0.5]
    df_pca = df_pca[df_pca['Connection_prob'] != 0.75]
    df_pca.loc[df_pca['RND'] == False, 'Connection_prob'] = 'optimal'
    df_pca['l23_exp_var'] = df_pca['l23_exp_var'].apply(lambda x: np.cumsum(np.array(x)[:32]))
    df_pca['l5_exp_var'] = df_pca['l5_exp_var'].apply(lambda x: np.cumsum(np.array(x)[:32]))
    
    df_ = df_pca[df_pca['Connection_prob'].isin(['optimal', 1.0, 0.0])]
    
    def calc_stats(df, col):
        means = df.groupby('Connection_prob')[col].apply(lambda x: np.mean(np.vstack(x), axis=0))
        stderrs = df.groupby('Connection_prob')[col].apply(lambda x: np.std(np.vstack(x), axis=0) / np.sqrt(len(x)))
        return means, stderrs

    def convert_ndarray(arr):
        return np.vstack(arr.to_numpy())

    means_l23, stderrs_l23 = calc_stats(df_, 'l23_exp_var')
    means_l5, stderrs_l5 = calc_stats(df_, 'l5_exp_var')

    means_l23, stderrs_l23 = map(convert_ndarray, [means_l23, stderrs_l23])
    means_l5, stderrs_l5 = map(convert_ndarray, [means_l5, stderrs_l5])


    # reverse the order of the rows
    means_l23 = means_l23[::-1, :]
    stderrs_l23 = stderrs_l23[::-1, :]
    means_l5 = means_l5[::-1, :]
    stderrs_l5 = stderrs_l5[::-1, :]

    # plot the results
    labels=['optimal', r'random L5$\rightarrow$L2/3 feedback', r'no L5$\rightarrow$L2/3 feedback']
    fig, ax = plt.subplots(2,1 , figsize=(3, 4))
    # add gray and rak gray colors for each sparsity
    colors = ['steelblue', '#5c5b5b', '#5c5b5b']
    line_styles = ['-', '-.', ':']
    for i in range(3):
        ax[0].errorbar(np.arange(1, means_l23.shape[1]+1), means_l23[i, :].T, yerr=stderrs_l23[i, :].T, label=labels[i], color=colors[i], linestyle=line_styles[i] )
        ax[0].set_title('L2/3', fontsize=8)
        ax[0].set_xlabel('principal component', fontsize=8)
        ax[0].set_ylabel('explained variance', fontsize=8)
        # remove spines
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        # set ticks fontsize
        ax[0].tick_params(axis='both', which='major', labelsize=8)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)
        
        ax[1].errorbar(np.arange(1, means_l5.shape[1]+1), means_l5[i, :].T, yerr=stderrs_l5[i, :].T, label=labels[i], color=colors[i], linestyle=line_styles[i] )
        ax[1].set_title('L5', fontsize=8)
        ax[1].set_xlabel('principal component', fontsize=8)
        ax[1].set_ylabel('explained variance', fontsize=8)
        # remove spines
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        # set ticks fontsize
        ax[1].tick_params(axis='both', which='major', labelsize=8)
        ax[1].tick_params(axis='both', which='minor', labelsize=8)
        

    ax[0].legend(title='connection probability', loc='upper right', bbox_to_anchor=(1.3, 1.), fontsize=8, title_fontsize=8)
    ax[1].legend(title='connection probability', loc='upper right', bbox_to_anchor=(1.3, 1.), fontsize=8, title_fontsize=8)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
        
        
def plot_activation_pca(run_arg,model_name, task_name, name,ckpt_dir_dict, save_fig=False):


    l5_out_all, l23_out_all, _, x1_label_all, x2_label_all, td_all = run_eval(model_name=model_name, task_name=task_name, args=run_arg,ckpt_dir_dict=ckpt_dir_dict, device=torch.device('cpu'))
    pca_l23 = PCA(n_components=2).fit(l23_out_all)
    l23_data_pca = pca_l23.transform(l23_out_all)
    
    pca_l5 = PCA(n_components=2).fit(l5_out_all)
    l5_data_pca = pca_l5.transform(l5_out_all)
    
    fig, ax = plt.subplots(2, 1, figsize=(2, 4))
    markers_ = ['o', 'v', 's']
    markers_index= x2_label_all-x1_label_all
    markers = [markers_[i] for i in markers_index]
    # list of 10 colors for a 10-class classification problem
    colors_ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [colors_[i] for i in x2_label_all]
    for i in range(l23_data_pca.shape[0]//2):
        ax[0].scatter(l23_data_pca[i, 0], l23_data_pca[i, 1], c=colors[i], s=5, marker= markers[i], alpha=0.6)
        ax[1].scatter(l5_data_pca[i, 0], l5_data_pca[i, 1], c=colors[i], s=5, marker= markers[i], alpha=0.6)
        
    ax[0].set_title('L2/3', fontsize=config.title_fontsize)
    ax[1].set_title('L5', fontsize=config.title_fontsize)
    ax[0].set_xlabel('principal component 1', fontsize=config.axis_fontsize)
    ax[1].set_ylabel('principal component 2', fontsize= config.axis_fontsize)
    # set ticks fontsize
    ax[0].tick_params(axis='both', which='major', labelsize=config.tick_fontsize)
    ax[0].tick_params(axis='both', which='minor', labelsize=config.tick_fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=config.tick_fontsize)
    ax[1].tick_params(axis='both', which='minor', labelsize=config.tick_fontsize)


    # # rempve spines
    sns.despine()
    
    # # show legend for unique colors

    # Your data
    unique_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    patches = [mpatches.Patch(color=colors_[i], label=str(unique_labels[i])) for i in range(len(unique_labels))]

    # Create a legend with specified fontsize and handle length
    digit_legend = plt.legend(handles=patches, title='gabor', loc='upper right', fontsize='small', handlelength=1)

    # Manually set the size of the legend patches
    for patch in digit_legend.get_patches():
        patch.set_height(2)  # Set the height of each patch
        patch.set_width(4)   # Set the width of each patch
        # add legend for unique markers
        

    # colors = ['#5fa6d1ff', 'gray','#e9724cff' ]
    gray_circle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                            markersize=4, label='0')
    blue_triangle = mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
                            markersize=4, label='-1')
    orange_dimoand = mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                            markersize=4, label='+1')

    marker_legend =plt.legend(handles=[blue_triangle, gray_circle, orange_dimoand], title='top-down')
    # add both legends back
    plt.gca().add_artist(digit_legend)
    plt.gca().add_artist(marker_legend)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'saved_figs/{name}', dpi=300)
    else:
        print('not saving fig')
        
        
        
        

        
        
def lifetime_kurtosis_neuron(r):
    M = r.shape[0] # number of stimuli
    r_bar = np.mean(r, axis=0)
    sigma = np.std(r, axis=0)
    sum_m = np.sum(((r - r_bar) / sigma)**4, axis=0)
    Kl = (sum_m / M) - 3
    return Kl

def lifetime_kurtosis_population(r):
    N = r.shape[1] # number of neuron
    Kps = []
    r_bar = np.mean(r, axis=0)
    sigma = np.std(r, axis=0)
        
    for f in range(r.shape[0]):
        r_f = r[f]
        sum_n = np.sum(((r_f - r_bar) / sigma)**4)
        Kp = (sum_n / N) - 3
        Kps.append(Kp)
    return np.mean(Kp)

def TR_measure(r):
    N = r.shape[1] # number of neuron
    STs = []
    for f in range(r.shape[0]):
        r_f = r[f]
        
        ST_num = np.sum(r_f / N)**2
        ST_denom = np.sum(r_f**2 / N)
        ST = 1 - (ST_num / ST_denom)
        STs.append(ST)
    return np.mean(STs)