import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 32
EPOCHS = 10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # 1, 28, 28 -> 32, 28, 28
        self.pool1 = nn.MaxPool2d(2) # 32, 14, 14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 64, 14, 14
        self.pool2 = nn.MaxPool2d(2) # 64, 7, 7

        self.dense1 = nn.Linear(64 * 7 * 7, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(128, 10)

        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='relu')
        with torch.no_grad():
            self.dense3.weight *= 0.1

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.bn1(self.dense1(x)))
        x = self.dropout1(x)
        x = self.dense3(x)
        return x

train_data = np.load("MNISTdata/mnist_train.npy")

X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_train = X_train / 255.0

test_data = np.load("MNISTdata/mnist_test.npy")
X_test, y_test = test_data[:, 1:], test_data[:, 0]  
X_test = X_test / 255.0

train_dataset = TensorDataset(torch.Tensor(X_train).view(-1, 1, 28, 28), torch.Tensor(y_train).to(torch.int64))
test_dataset = TensorDataset(torch.Tensor(X_test).view(-1, 1, 28, 28), torch.Tensor(y_test).to(torch.int64))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(EPOCHS):
    print(f"Epoch #{e + 1} --------------------")
    model.train()

    for idx, (input, label) in enumerate(train_loader):
        optimizer.zero_grad()

        logits = model.forward(input)

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