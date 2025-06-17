import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(EPOCHS):
    print(f"Epoch #{e + 1} --------------------")
    model.train()

    for idx, (input, label) in enumerate(train_loader):
        optimizer.zero_grad()

        logits = model(input)

        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            loss, current = loss.item(), idx * BATCH_SIZE + len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataset):>5d}]")
    
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for input, label in test_loader:
            logits = model(input)
            test_loss += F.cross_entropy(logits, label).item()
            correct += (logits.argmax(1) == label).type(torch.float).sum().item()

        test_loss /= len(test_loader)
        correct /= len(test_dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f}\n")