from torch.utils.data import Dataset

class myDataset(Dataset):

    def __init__(self, feattrain, feattest, labeltrain, labeltest, train=True):
        self.train = train
        self.feattrain = feattrain
        self.feattest = feattest
        self.labeltrain = labeltrain
        self.labeltest = labeltest

    def __getitem__(self, index):

        if self.train:
            return self.feattrain[index], self.labeltrain[index]
        else:
            return self.feattest[index], self.labeltest[index]

    def __len__(self):

        if self.train:
            return len(self.labeltrain)
        else:
            return len(self.labeltest)









