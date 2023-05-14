import os
import pandas as pd
from PIL import Image
import numpy as np
import torch as th
import getDataset
import json

#preprocessing and loading the dataset
class SiameseDataset():
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        self.train_df = pd.read_csv(training_csv,
                                    dtype={"car1": object, "car2": object, "class1": object, "class2": object},
                                    header=0)
        self.train_df.columns = ["image1", "image2", "label", "class1", "class2"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0] + ".jpg")
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1] + ".jpg")
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        number = self.train_df.iloc[index, 3]
        zeros = [0] * 250
        zeros[number] = 1
        class1 = torch.Tensor(zeros) 
        return img0, img1, th.from_numpy(np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)), class1

    def __len__(self):
        return len(self.train_df)


def getSiameseDataset(training_csv=None, training_dir=None, transform=None):
    return SiameseDataset(training_csv, training_dir, transform)