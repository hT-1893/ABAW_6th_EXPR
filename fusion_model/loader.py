from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import numpy as np

class FeatureDataset(Dataset):

    def __init__(self, feature_file, label_file):
        self.features = np.load(feature_file)
        self.labels = np.load(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return torch.from_numpy(feature), label

def get_train_dataloader(anno_dir, img_dir, batch_size):
    dataset = FeatureDataset(anno_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return dataloader

def get_val_dataloader(anno_dir, img_dir, batch_size):
    dataset = FeatureDataset(anno_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader