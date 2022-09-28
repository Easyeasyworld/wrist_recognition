import torch


class EMGdataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)
