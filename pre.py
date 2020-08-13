from os import path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def build_data(data_set,batch_size=20):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), #把灰度范围从0-255变换到0-1之间
        #transforms.Normalize(RGB_mean, RGB_std) #image=(image-mean)/std 
    ])

    data_dir = path.join('./101_Categories', data_set)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transform)
    dataloadder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloadder

def compute_rgb_mean_std(data_set='train', batch_size=4):
    _, loader = build_data(data_set, batch_size)
    mean, std, sample_num = 0., 0., 0.
    for data, _ in loader:
        batch_num = data.size(0)
        data = data.view(batch_num, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        sample_num += batch_num
    
    mean /= sample_num
    std /= sample_num
    return mean, std

if __name__ == '__main__':
    mean, std = compute_rgb_mean_std(batch_size=16)
    print(mean)
    print(std)