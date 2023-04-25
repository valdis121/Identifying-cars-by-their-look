from multiprocessing import freeze_support

import torch
from torch import nn
import torch as th
from torch.utils.data import DataLoader
from torchgen.context import F
from getDataset import getSiameseDataset
from torchvision import transforms

training_csv = 'result.csv'
training_dir = '../VehicleID_V1.0/image/'
siamese_dataset = getSiameseDataset(training_csv, training_dir,
                                        transform=transforms.Compose([transforms.Resize((105,105)),
                                                                      transforms.Grayscale(),
                                                                      transforms.ToTensor()
                                                                      ]))

# create a siamese network



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
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

            nn.Linear(128, 64))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                          2))

        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset, num_workers=8, batch_size=128, shuffle=True)
# Declare Siamese Network
net = SiameseNetwork().cuda()
#net = SiameseNetwork()
# Decalre Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)


def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(1, 5):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            #img0, img1, label = img0, img1, label
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i%10 == 0:
                print("Epoch {}\n Current loss {}\n Image {}/{}\n".format(epoch, loss_contrastive.item(), i, len(train_dataloader)))
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    return net

    # set the device to cuda
if __name__ == '__main__':
    device = torch.device('cuda' if th.cuda.is_available() else 'cpu')
    model = train()
    torch.save(model.state_dict(), "model.pt")
    print("Model Saved Successfully")