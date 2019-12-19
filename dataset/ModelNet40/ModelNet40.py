import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import provider
import config
from .. import provider
from .. import pointfly as pf

class ModelNet40(Dataset):
    """
        Dataset for train and validation set
    """

    def __init__(self, root_dir, type):
        assert type in ["train","val","test"]
        super(ModelNet40,self).__init__()
        self.root_dir = root_dir
        self.train_files = provider.getDataFiles(self.root_dir)
        data = []
        label = []
        for i in range(len(self.train_files)):
            raw_data = provider.loadDataFile(self.train_files[i])
            for pt in raw_data[0]:
                data.append(pt)
            for lb in raw_data[1]:
                label.append(lb)
        self.__data = torch.Tensor(np.array(data))
        self.__label = torch.Tensor(np.array(label))

    def __getitem__(self, index):
        data = self.__data[index]
        return data, self.__label[index]

    def __len__(self):
        return len(self.__label)

