import torch
from torch import nn
from PIL import Image
import torch as th
from torch.utils.data import DataLoader
from torchgen.context import F
from torchvision import transforms
from collections import defaultdict

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            # Удаляем последний слой, который вычисляет расстояния

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input):
        # forward pass
        output = self.forward_once(input)
        return output

state_dict = torch.load('temp_net30.pt')
model = MyNetwork()
model.load_state_dict(state_dict)

pathToLabels = '../VehicleID_V1.0/train_test_split/test_list_800.txt'
nameOfResult = 'val.csv'
result = open(nameOfResult, 'w')
labels = open(pathToLabels, 'r')

for line in labels:
    strings = line.split(' ')
    car = ''.join(e for e in strings[0] if e.isalnum())
    id = ''.join(e for e in strings[1] if e.isalnum())
    image = Image.open('../VehicleID_V1.0/image/{}.jpg'.format(car))
    image = image.resize((105, 105))
    image = image.convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model.forward_once(img_tensor)
        result.write("{},{}\n".format(output, id))



#euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

