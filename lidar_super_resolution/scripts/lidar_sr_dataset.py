#!/usr/bin/env python3

from data import *
import torch.utils.data as data
class LiDARSRDataset(data.Dataset):
    def __init__(self):
        self.train_x, self.train_y = load_train_data()

    def __getitem__(self, index):
        index = index % self.__len__()
        x = self.train_x[index, :, :, :]
        y = self.train_y[index, :, :, :]
        return x, y

    def __len__(self):
        return self.train_x.shape[0]

if __name__ == "__main__":
    dataset = LiDARSRDataset()
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape)
    print(y.shape)