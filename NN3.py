import zipfile   
import os  
import time      
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import argparse
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from collections import defaultdict
import csv 
import random
import pandas as pd
from PIL import Image
import torch as th 
from getDataset2 import getSiameseDataset 
from torch.utils.data import DataLoader

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 250)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    
class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"): 
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone)) 
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True) 
        out_features = list(self.backbone.modules())[-1].out_features 
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(1000, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(out_features, 250)
        self.fc1 = nn.Linear(out_features, 1000)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(1000, 500)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, 250)

        self.batch = nn.BatchNorm1d(1000)

    def forward(self, img1, img2): 
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2) 
        output1 = self.head(feat1)
        output2 = self.head(feat2)
        output = torch.nn.functional.pairwise_distance(output1, output2)
        classify = self.fc1(feat1)
        classify = self.sigmoid(classify)
        classify = self.batch(classify)
        classify = self.fc2(classify)
        classify = self.relu(classify)
        classify = self.fc3(classify) 

        return output, classify

    def forward_once(self, img):
        feat = self.backbone(img)
        output = self.head(feat)
        return output

    def classify(self, features):
        output = self.classifier(features)
        return output

def formDictionary2(csvPah='VehicleID_V1.0/train_test_split/train_list.txt'):
    with open(csvPah, 'r') as csv_file:
        my_dict = {}
        for line in csv_file:
            elems = line.split()
            my_dict[elems[0]] = elems[1]

    return my_dict


def formDictinary(csvPah='model_attr_converted.csv'):
    with open(csvPah, 'r') as csv_file:
        my_dict = {}
        for line in csv_file:
            elems = line.split()
            my_dict[elems[0]] = elems[1]

    return my_dict


def filerCSV(path, data, data_attr):
    with open(path, 'r') as input_file, open('filtred_test.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        for line in input_file:
            line = line.strip()
            fields = line.split()
            if data.get(fields[1]) is None:
                continue
            writer.writerow(fields)

def formFilter(path="VehicleID_V1.0/train_test_split/train_list.txt"):
    result = open("model_attr_converted.csv", 'w') 
    result.close()
    data = formDictinary("VehicleID_V1.0/attribute/model_attr.txt")
    data_attr = formDictionary2()
    filerCSV(path, data, data_attr)

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def forward(self, euc_dist, flag, res_class, target_class): 
        loss1 = torch.mean((1 - flag) * torch.pow(euc_dist, 2) +
                           (flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))
        criterion = nn.CrossEntropyLoss()
        loss2 = criterion(res_class, target_class)
        loss = 0.5 * loss1 + (1 - 0.5) * loss2
        return loss, loss1, loss2

    

if __name__ == '__main__': 
    formFilter()
    ids1 = dict()
    with open('filtred_test.csv', 'r') as f:
        for line in f:
            strings = line.split(',')
            ids1[strings[1]] = False
    list_ids = list(ids1.keys()) 
    indexes = random.sample(list_ids, int(len(list_ids)*0.1))
    for i in indexes:
        ids1[i] = True
    result1 = open('for_val.csv', 'w')
    result2 = open('for_result.csv', 'w')
    val_counter = 0
    result_counter = 0
    with open('filtred_test.csv', 'r') as csv_file:  
        for line in csv_file:
            strings = line.split(',') 
            if ids1[strings[1]]:
                result1.write(line)
                val_counter+=1
            else:
                result2.write(line)
                result_counter+=1

    print("result = " + str(result_counter))
    print("val = " + str(val_counter))
    def def_value1():
        return "Nothing"


    def def_value2():
        return []


    data = formDictinary('VehicleID_V1.0/attribute/model_attr.txt')
    minCost = 0
    maxCost = 1
    # pathToLabels = 'datset2/Dataset/train_test_split/test_list_800.txt' 
    pathToLabels = 'for_result.csv'

    nameOfResult = 'result.csv'
    numberOfSamplesOnOneCar = 5

    labels = open(pathToLabels, 'r')
    result = open(nameOfResult, 'w')

    result.write("car1,car2,distance,vector1,vector2\n")

    carsId = defaultdict(def_value1)
    idCars = defaultdict(def_value2)

    for line in labels:
        strings = line.split(',')
        car = ''.join(e for e in strings[0] if e.isalnum())
        id = ''.join(e for e in strings[1] if e.isalnum())
        carsId[car] = id
        idCars[id].append(car)

    n = 0
    cars = list(carsId.keys())
    ids = list(idCars.keys())

    for i in ids:
        car = idCars[i][0]
        for x in idCars[i]:
            if car == x:
                continue
            n += 1
            dis = minCost
            carList1, carList2 = data[carsId[car]], data[carsId[car]]
            result.write("{},{},{},{},{}\n".format(car, x, dis, carList1, carList2))

    for x in range(len(cars)):
        if x + 1 >= len(cars):
            break
        randomList = random.sample(range(0, len(cars)), numberOfSamplesOnOneCar)
        for y in randomList:
            car1 = cars[x]
            car2 = cars[y]

            if car1 == car2:
                continue

            dis = maxCost
            n += 1
            if carsId[car1] == carsId[car2]:
                continue
            carList1, carList2 = data[carsId[car1]], data[carsId[car2]]
            result.write("{},{},{},{},{}\n".format(car1, car2, dis, carList1, carList2))

    labels.close()
    result.close()
    result2.close()

    print("Num of samples = {}".format(n)) 


    minCost = 0
    maxCost = 1
    # pathToLabels = 'datset2/Dataset/train_test_split/test_list_800.txt' 
    pathToLabels = 'for_val.csv'

    nameOfResult = 'val.csv'
    numberOfSamplesOnOneCar = 5

    labels = open(pathToLabels, 'r')
    result = open(nameOfResult, 'w')

    result.write("car1,car2,distance,vector1,vector2\n")

    carsId = defaultdict(def_value1)
    idCars = defaultdict(def_value2)

    for line in labels:
        strings = line.split(',')
        car = ''.join(e for e in strings[0] if e.isalnum())
        id = ''.join(e for e in strings[1] if e.isalnum())
        carsId[car] = id
        idCars[id].append(car)

    n = 0
    cars = list(carsId.keys())
    ids = list(idCars.keys())

    for i in ids:
        car = idCars[i][0]
        for x in idCars[i]:
            if car == x:
                continue
            n += 1
            dis = minCost
            carList1, carList2 = data[carsId[car]], data[carsId[car]]
            result.write("{},{},{},{},{}\n".format(car, x, dis, carList1, carList2))

    for x in range(len(cars)):
        if x + 1 >= len(cars):
            break
        randomList = random.sample(range(0, len(cars)), numberOfSamplesOnOneCar)
        for y in randomList:
            car1 = cars[x]
            car2 = cars[y]

            if car1 == car2:
                continue

            dis = maxCost
            n += 1
            if carsId[car1] == carsId[car2]:
                continue
            carList1, carList2 = data[carsId[car1]], data[carsId[car2]]
            result.write("{},{},{},{},{}\n".format(car1, car2, dis, carList1, carList2))

    labels.close()
    result.close()
    result2.close()

    print("Num of samples = {}".format(n))


    training_csv = 'result.csv'
    val_csv = 'val.csv'
    training_dir = 'VehicleID_V1.0/image'
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

    train_dataset = getSiameseDataset(training_csv, training_dir, transform=transform1)
    val_dataset = getSiameseDataset(val_csv, training_dir, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=128, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    model = SiameseNetwork(backbone="resnet18")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()
    criterion2 = nn.CrossEntropyLoss()

    writer = SummaryWriter("summary")

    best_val = 10000000000

    model = SiameseNetwork(backbone="resnet18")
    model.to(device)
    criterion2 = ContrastiveLoss()

    for epoch in range(100):
        print("[{} / {}]".format(epoch, 100))
        model.train()
        print(device)
        losses = []
        losses1= []
        losses2= []

        correct = 0
        total = 0
        iterator = True
        counter = 0
        # Training Loop Start
        for img1, img2, y, classes in train_dataloader:
            img1, img2, y, classes = map(lambda x: x.to(device), [img1, img2, y, classes])

            prob, output_classify = model(img1, img2)
            loss, loss1, loss2 = criterion2(prob, y, output_classify, target_class=classes)
            
            counter+=1
            optimizer.zero_grad()
            losses.append(loss.item())
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            total += len(y)
            
            if(iterator):
                loss1.backward()
                optimizer.step() 
                iterator = False
            else:
                loss2.backward()
                optimizer.step()
                iterator = True
            if counter%5 == 0: 
                print("-----")
                print("\tTraining1: Loss={:.2f}\t".format(sum(losses1) / len(losses1)))
                print("\tTraining2: Loss={:.2f}\t".format(sum(losses2) / len(losses2)))
                print("\tTraining: Loss={:.2f}\t".format(sum(losses) / len(losses)))
                print("-----")
            
            if (total / 128) % 100 == 0:
                print("Actually processed {}/{}".format(total, len(train_dataloader) * 128))
                print("\tTraining: Loss={:.2f}\t".format(sum(losses) / len(losses)))

        writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)

        print("\tTraining: Loss={:.2f}\t".format(sum(losses) / len(losses)))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for img1, img2, y, classes in val_dataloader:
            img1, img2, y, classes = map(lambda x: x.to(device), [img1, img2, y, classes])

            prob, output_classify = model(img1, img2)
            loss, loss1, loss2 = criterion2(prob, y, output_classify, target_class=classes)

            losses.append(loss.item())
            total += len(y)

        val_loss = sum(losses) / max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)

        print("\tValidation: Loss={:.2f}\t".format(val_loss))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        name = "drive/MyDrive/2_{}_{}_{}.pth".format(str(sum(losses) / len(losses)), val_loss, epoch)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "backbone": "resnet18",
                "optimizer_state_dict": optimizer.state_dict()
            },
            name
        )
        if val_loss < best_val:
            name = "drive/MyDrive/2_Best{}_{}_{}.pth".format(str(sum(losses) / len(losses)), val_loss, epoch)

            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": "resnet18",
                    "optimizer_state_dict": optimizer.state_dict()
                },
                name
            )