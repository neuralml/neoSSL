import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticVisualFlowDataset(Dataset):
    def __init__(self, num_samples=1000, dimensions=8, max_value=1,  var_scaler=0.01, speed_func="random", num_speeds=11):
        assert speed_func in ["random", "random_walk", "mismatch"]
        assert dimensions > 0, "dimensions must be greater than 0"
        assert num_speeds % 10 == 1, "num_speeds must be 10n + 1"
        self.num_samples = num_samples
        self.dimensions = dimensions
        self.max_value = max_value

        self.var_scaler = var_scaler
        self.speed_func = speed_func
        self.num_speeds = num_speeds

        self._get_speeds()
        self._generate_visual_flow()

    def __len__(self):
        return self.num_samples - 1
    
    def _get_speeds(self):
        # get the possible values of speeds from 0 to max_value given num_speeds
        speeds = np.linspace(0, self.max_value, self.num_speeds)
        print(f'{speeds=}')
        if self.speed_func == "random":
            self.speeds = np.random.choice(speeds, self.num_samples)
        elif self.speed_func == "random_walk":
            self.speeds = np.zeros(self.num_samples)
            idx = 0
            self.speeds[idx] = speeds[idx]
            for i in range(1, self.num_samples):
                steps = np.random.choice([-1, 0, 1], self.num_samples)
                idx = np.clip(idx + steps[i], 0, len(speeds)-1)
                self.speeds[i] = speeds[idx]
        elif self.speed_func == "mismatch":
            assert self.num_samples % len(np.unique(speeds)) == 0, "num_samples must be divisible by num_speeds"
            each_speed = self.num_samples // len(np.unique(speeds))
            self.speeds = np.zeros(each_speed * len(np.unique(speeds)))
            for i,s in enumerate(speeds):
                self.speeds[i*each_speed:(i+1)*each_speed] = s


    def _generate_visual_flow(self):
        # map speeds to be between 0 and 1
        speeds = self.speeds / np.max(self.speeds)
        max_var = (1 / self.num_speeds) * self.max_value
        var = self.var_scaler * max_var
        noise = np.random.rand(self.num_samples, self.dimensions) * var

        self.visual_flow = speeds[:, None] * self.max_value + noise
        
        
    def __getitem__(self, idx):
        speeds_t = self.speeds[idx:idx+1]
        speeds_tp1 = self.speeds[idx+1: idx+2]
        
        visual_flow_t = self.visual_flow[idx]
        visual_flow_tp1 = self.visual_flow[idx+1]
        return visual_flow_t, visual_flow_tp1, speeds_t, speeds_tp1
    