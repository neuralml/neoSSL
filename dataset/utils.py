import torch
from torch.utils.data import Dataset

import numpy as np

class NoisyDataset(Dataset):
    def __init__(self, x1, x2, td, noise = 0.0):
        self.x1 = x1
        self.x2 = x2
        self.td = td
        self.noise = noise
        self.indexer = len(x1)

    def __getitem__(self, idx):
        return self.x1[idx%self.indexer] + self.noise*torch.randn_like(self.x1[idx%self.indexer])  , self.x2[idx%self.indexer] + self.noise*torch.randn_like(self.x2[idx%self.indexer]), self.td[idx%self.indexer], self.x1[idx%self.indexer], self.x2[idx%self.indexer]

    def __len__(self):
        return len(self.x1)
    
    
class OcclusionDataset(Dataset):
    def __init__(self, x1, x2, td, x1_label, x2_label, occlusion_size = 0.0, moving=False):
        self.x1 = x1
        self.x2 = x2
        self.td = td
        self.occlusion_size = occlusion_size
        self.x1_label = x1_label
        self.x2_label = x2_label
        self.indexer = len(x1)
        self.moving = moving
        
    def __getitem__(self, idx):
        canvas_x1 = torch.randn(28,28) * 0.02
        canvas_x2 = canvas_x1.clone()
        img_size = self.x1[idx%self.indexer].shape[0]
        x1_s, y1_s = np.random.randint(5, 23-img_size, size=2)
        if self.moving:
            x_step, y_step = np.random.choice([0, 5], size=2) # stochasticity in the movement of patches
        else:
            x_step, y_step = 0, 0
        canvas_x1[x1_s:x1_s+img_size, y1_s:y1_s+img_size] = self.x1[idx%self.indexer]
        canvas_x2[x1_s+x_step:x1_s+img_size+x_step, y1_s+y_step:y1_s+img_size+y_step] = self.x2[idx%self.indexer]
        
        if self.occlusion_size > 0:
            mask = torch.ones_like(canvas_x1)

            # randomly select a random squared region of size self.occlusion_size*self.x1[idx%self.indexer].shape[0] x self.occlusion_size*self.x1[idx%self.indexer].shape[1] and set it to 0
            mask_region = int(self.occlusion_size * canvas_x1.shape[0])
            x_y_starts_x = np.random.randint(0, canvas_x1.shape[0] - mask_region, size=2)

            mask[x_y_starts_x[0]:x_y_starts_x[0]+mask_region, x_y_starts_x[1]:x_y_starts_x[1]+mask_region] = 0.0

            # apply the mask to the image
            x1 = canvas_x1
            if np.random.rand() > 0.5: # randomly apply mask to x2
                x2 = canvas_x2 * mask
            else:
                x2 = canvas_x2
            return x1, x2, self.td[idx%self.indexer], self.x1_label[idx%self.indexer], self.x2_label[idx%self.indexer]

        return self.x1[idx%self.indexer] ,self.x2[idx%self.indexer] , self.td[idx%self.indexer], self.x1_label[idx%self.indexer], self.x2_label[idx%self.indexer]

    def __len__(self):
        return len(self.x1)
    
    
    
    
    

    