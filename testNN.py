import torch
from torch import nn
from PIL import Image
import torch as th
from torch.utils.data import DataLoader
from torchgen.context import F
from torchvision import transforms
from collections import defaultdict
from torchvision import models
import random

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
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
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
        combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(combined_features)
        return output

state_dict = torch.load('epoch_14.pth')
model = SiameseNetwork()
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
model.cuda()
pathToLabels = '../VehicleID_V1.0/train_test_split/test_list_800.txt'
nameOfResult = 'res.csv'
res = open(nameOfResult, 'w')
labels = open(pathToLabels, 'r')
feed_shape = [3, 224, 224]
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(feed_shape[1:])
    ])
t = list(labels)
test = []
correct1=0
correct5=0
k=0
for m in t:
    print('Proccesed {}/{}\n'.format(k, len(t)))
    k += 1
    strings = m.split(' ')
    car_main = ''.join(e for e in strings[0] if e.isalnum())
    id_main = ''.join(e for e in strings[1] if e.isalnum())
    image = Image.open('../VehicleID_V1.0/image/{}.jpg'.format(car_main))
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    result = []
    for line in t:
        strings = line.split(' ')
        car = ''.join(e for e in strings[0] if e.isalnum())
        if car == car_main:
            continue
        id = ''.join(e for e in strings[1] if e.isalnum())
        image2 = Image.open('../VehicleID_V1.0/image/{}.jpg'.format(car))
        img_tensor2 = transform(image2)
        img_tensor2 = img_tensor2.unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor2.cuda(), img_tensor.cuda())
            result.append((float(output.cpu().numpy()[0]), id))
    def sort_col(i):
        return i[0]
    result.sort(key=sort_col)
    if result[0][1]==id_main:
        correct1+=1
    for i in range(5):
        if result[i][1]==id_main:
            correct5+=1
            break

n1 = (correct1*100)/len(t)
n5 = (correct5*100)/len(t)
print("Hit 1 = {}%".format(n1))
print("Hit 5 = {}%".format(n5))

