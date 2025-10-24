import torch
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    '''
    iris dataset custom class
    '''
    def __init__(self, train_x, train_y):
        super().__init__()
        self.x = torch.from_numpy(train_x)
        self.y = torch.from_numpy(train_y)

        # Most PyTorch losses require targets in long tensor type
        self.y = self.y.type(torch.LongTensor)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
