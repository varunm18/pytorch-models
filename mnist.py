import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from net import Net

BATCH_SIZE = 64
EPOCHS = 20

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(128, 10)

        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity='relu')
        with torch.no_grad():
            self.dense3.weight *= 0.1

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.bn1(self.dense1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.dense2(x)))
        x = self.dropout2(x)
        x = self.dense3(x)
        return x

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

model = Model()
net = Net(model, F.cross_entropy, torch.optim.Adam(model.parameters(), lr=1e-3), 
          train_dataset, test_dataset, BATCH_SIZE)

for e in range(EPOCHS):
    net.train(e)
    net.test()

if(input("Save? ") in ['Y', 'y']):
    torch.save(model.state_dict(), f"models/mnist/{input('Save As: ')}")