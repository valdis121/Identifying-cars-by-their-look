import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

class SiameseDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_path, img_dir, batch_size, img_size):
        self.df = pd.read_csv(csv_path, dtype={'car1': object, 'car2':object})
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop=True)
        batch_df = batch_df.iloc[np.random.permutation(len(batch_df))]  # перемешиваем индексы строк
        batch_images1 = []
        batch_images2 = []
        batch_labels = []
        for i, row in batch_df.iterrows():
            img1 = cv2.imread("{}{}.jpg".format(self.img_dir, row['car1']))
            img1 = cv2.resize(img1, (self.img_size, self.img_size))
            batch_images1.append(img1)

            img2 = cv2.imread("{}{}.jpg".format(self.img_dir, row['car2']))
            img2 = cv2.resize(img2, (self.img_size, self.img_size))
            batch_images2.append(img2)

            batch_labels.append(row['distance'])
        return [np.array(batch_images1), np.array(batch_images2)], np.array(batch_labels)

    def get_batch_pairs(self, batch_size):
        batch_df = self.df.sample(batch_size)
        batch_images1 = []
        batch_images2 = []
        batch_labels = []
        for i, row in batch_df.iterrows():
            img1 = cv2.imread("{}{}.jpg".format(self.img_dir, row['car1']))
            img1 = cv2.resize(img1, (self.img_size, self.img_size))
            batch_images1.append(img1)

            img2 = cv2.imread("{}{}.jpg".format(self.img_dir, row['car2']))
            img2 = cv2.resize(img2, (self.img_size, self.img_size))
            batch_images2.append(img2)

            batch_labels.append(row['distance'])
        return [np.array(batch_images1), np.array(batch_images2)], np.array(batch_labels)


def getSiameseDataset(csv_path, img_dir, batch_size, img_size):
    return SiameseDataset(csv_path, img_dir, batch_size, img_size)
