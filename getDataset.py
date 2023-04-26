import os

import pandas as pd
from PIL import Image
import numpy as np
import torch as th
import getDataset

#preprocessing and loading the dataset
class SiameseDataset():
    def __init__(self,training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv, dtype={"car1": object, "car2": object})
        self.train_df.columns =["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform


    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0] + ".jpg")
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1] + ".jpg")
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, th.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    def __len__(self):
        return len(self.train_df)

def getSiameseDataset(training_csv=None, training_dir=None, transform=None):
    return SiameseDataset(training_csv, training_dir, transform)

