import numpy as np
import torch
from torch.utils.data import Dataset

import sys
import config
from .. import data_utils
from .. import pointfly as pf

class ShapeNetParts(Dataset):
    """
        Dataset for train and validation set
    """

    def __init__(self, root_dir):
        super(ShapeNetParts,self).__init__()
        self.root_dir = root_dir

        self.__points, _, self.__point_nums, self.__labels_seg, _ = data_utils.load_seg(self.root_dir)
        self.__points=torch.Tensor(pf.global_norm(self.__points))
        self.__points_nums+=100

        print(self.__points.shape)
        print(self.__labels_seg.shape)

    def __getitem__(self, index):
        return self.__points[index], self.__labels_seg[index], self.__point_nums[index]

    def __len__(self):
        return self.__points.shape[0]

