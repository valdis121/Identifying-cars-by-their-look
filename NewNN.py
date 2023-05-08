import torch
import torch.nn as nn
import torch.nn.functional as F
from getDataset import getSiameseDataset
from torchvision import models
import os
import argparse
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.
            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head.
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(1000, 5),
            nn.ReLU(),
        )

    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.
            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]
            where b = batch size
            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # Multiply (element-wise) the feature vectors of the two images together,
        # to generate a combined feature vector representing the similarity between the two.

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output1 = self.head(feat1)
        output2 = self.head(feat2)
        output = torch.nn.functional.pairwise_distance(output1, output2)
        return output
    def forward_once(self, img):
        feat = self.backbone(img)
        output = self.head(feat)
        return output


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def forward(self, euc_dist, flag):
        # flag = 0 means y1 and y2 are supposed to be same
        # flag = 1 means y1 and y2 are supposed to be different
        loss = torch.mean((1 - flag) * torch.pow(euc_dist, 2) +
                      (flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))

        return loss
if __name__ == '__main__':

    training_csv = 'result.csv'
    val_csv = 'val.csv'
    training_dir = '../VehicleID_V1.0/image/'
    feed_shape = [3, 224, 224]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(feed_shape[1:])
    ])
    transform1 = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(feed_shape[1:])
    ])
    train_dataset = getSiameseDataset(training_csv, training_dir,
                                        transform=transform1)
    val_dataset = getSiameseDataset(val_csv, training_dir,
                                      transform=transform)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=8, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = SiameseNetwork(backbone="resnet18")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()

    writer = SummaryWriter("summary")

    best_val = 10000000000

    for epoch in range(100):
        print("[{} / {}]".format(epoch, 100))
        model.train()
        print(device)
        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for img1, img2, y in train_dataloader:

            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += len(y)
            if total%100 == 0:
                print("Actually processed {}/{}".format(total, len(train_dataloader)*8))
                print("\tTraining: Loss={:.2f}\t".format(sum(losses) / len(losses)))

        writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)

        print("\tTraining: Loss={:.2f}\t".format(sum(losses) / len(losses)))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for img1, img2, y in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            total += len(y)


        val_loss = sum(losses) / max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)

        print("\tValidation: Loss={:.2f}\t".format(val_loss))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": "resnet18",
                    "optimizer_state_dict": optimizer.state_dict()
                },
                "best.pth"
            )

            # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": "resnet18",
                    "optimizer_state_dict": optimizer.state_dict()
                },
                "epoch_{}.pth".format(epoch + 1)
            )