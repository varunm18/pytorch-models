import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from net import Net

BATCH_SIZE = 32
EPOCHS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # 1, 28, 28 -> 32, 28, 28
        self.pool1 = nn.MaxPool2d(2) # 32, 14, 14
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 64, 14, 14
        self.pool2 = nn.MaxPool2d(2) # 64, 7, 7
        self.bn2 = nn.BatchNorm2d(64)

        self.dense1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(128, 10)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity='relu')
        with torch.no_grad():
            self.dense3.weight *= 0.1

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.bn3(self.dense1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn4(self.dense2(x)))
        x = self.dropout2(x)
        x = self.dense3(x)
        return x

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

model = Model()
net = Net(model, F.cross_entropy, torch.optim.Adam(model.parameters(), lr=1e-3), 
          train_dataset, test_dataset, BATCH_SIZE)

for e in range(EPOCHS):
    net.train(e)
    net.test()

if(input("Save? ") in ['Y', 'y']):
    torch.save(model.state_dict(), f"models/fashion/{input('Save As: ')}")