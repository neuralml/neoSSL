import itertools
from skimage.filters import gabor_kernel
from skimage.transform import resize
from scipy.ndimage import convolve

import torch
from torch.utils.data import Dataset
import numpy as np

class GaborDatasetOrientation(Dataset):
    def __init__(self, gabor_mode, n_samples, topdown, image_size=28, noise=0):
        self.n_samples = n_samples - n_samples%10 # divisable by 10 (# digits)
        self.n_digit_samples = n_samples // 10
        assert gabor_mode in ['variable','exemplar']
        self.gabor_mode = gabor_mode
        self.fixed_canvas = np.random.rand(9*image_size, image_size).reshape(9,image_size,image_size)

        self.thetas = np.linspace(0., np.pi, 10)[:-1]
        self.base_freq = 0.2
        self.image_size = image_size
        self.images = self.gen_images()  
        self.noise=noise

        assert topdown in ['random_walk', 'deterministic']
        self.topdown = topdown
        self.steps = self.gen_steps()
        
        self.mean = np.mean([np.mean(imgs) for _,imgs in self.images.items()])
        self.std = np.mean([np.std(imgs) for _,imgs in self.images.items()])
        for k in self.images:
            self.images[k] = (self.images[k] - self.mean) / self.std

    def gen_steps(self):
        theta_vals = np.arange(0, len(self.thetas), 1)
        max_theta = max(theta_vals)
        min_theta = min(theta_vals)
        
        if self.topdown == 'random_walk':
            curr_theta = 0 
            vals = [curr_theta]
            for _ in range(self.n_samples):
                curr_theta += np.random.choice([1,0,-1])
                if curr_theta > max_theta:
                    curr_theta = max_theta
                elif curr_theta < min_theta:
                    curr_theta = min_theta
                vals.append(curr_theta)
            return np.array(vals)


        elif self.topdown == 'deterministic':
            topdown_vals = np.arange(0, len(self.thetas), 1)
            # repeate / tile topdown_vals to match n_samples
            return np.tile(topdown_vals, self.n_samples//len(topdown_vals)+1)[:self.n_samples]

    def create_gabor(self, size, theta, offset, class_label):
        
        # Convolve with a random image to create Gabor patch
        if self.gabor_mode == 'variable':
            frequency = self.base_freq + np.random.normal(0,0.1)
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=np.random.uniform(3,8), sigma_y=np.random.uniform(3,8), offset=offset))
            rand_img = (np.random.rand(size, size) + self.fixed_canvas[class_label]) / 2

            rand_img = self.normalise(rand_img)
        else:
            kernel = np.real(gabor_kernel(self.base_freq, theta=theta, sigma_x=10, sigma_y=10, offset=offset))
            rand_img = self.fixed_canvas[class_label]
        gabor_patch = convolve(rand_img, kernel)
        # Resize image to desired size
        gabor_patch_resized = resize(gabor_patch, (size, size))
        return gabor_patch_resized, kernel
    
    def gen_images(self):
        data = {}
        for i in range(self.n_samples):
            class_label = i%len(self.thetas)
            if class_label not in data:
                data[class_label] = []
            
            theta = self.thetas[class_label]
            gabor_image, kernel = self.create_gabor(size=self.image_size, theta=theta, offset=0, class_label=class_label)
            data[class_label].append(gabor_image)
        return data
    
    def normalise(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def __getitem__(self, key):
        td1 = self.steps[key]
        td2 = self.steps[key+1]

        x1 =  self.normalise(self.images[td1][np.random.randint(len(self.images[td1]))])
        x2 =  self.normalise(self.images[td2][np.random.randint(len(self.images[td2]))])

        if self.noise > 0:
            x1 = x1 + torch.rand_like(x1)*self.noise
            x2 = x2 + torch.rand_like(x2)*self.noise


        return torch.tensor(x1).float(), torch.tensor(x2).float(),torch.tensor(td2-td1), td1, td2


    def __len__(self):
        return self.n_samples-1 # to avoid out of index 
    