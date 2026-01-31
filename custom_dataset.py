import os
import cv2
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,root_dir,transforms = None):

        noise_dir = os.path.join(root_dir, "noise")
        label_dir = os.path.join(root_dir, "label")
        self.data_paths  = [os.path.join(noise_dir, f) for f in sorted(os.listdir(noise_dir))]
        self.label_paths = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))]
        self.transforms = transforms
        assert len(self.data_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.data_paths)} noise vs {len(self.label_paths)} label" # to ensure the index of label and noised matches

    def __getitem__(self,idx):
        img = cv2.imread(self.data_paths[idx])
        label = cv2.imread(self.label_paths[idx])

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if(self.transforms):
            img = self.transforms(img)
            label = self.transforms(label)
        return img, label
    
    def __len__(self):
        return len(self.data_paths)