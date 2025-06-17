import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam
from torch.utils.data import DataLoader
from collections.abc import Callable

class Net():
    def __init__(self, model, loss, optimizer, train_dataset, test_dataset, batch_size):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
    
    def train(self, epoch):
        print(f"Epoch #{epoch + 1} --------------------")
        self.model.train()

        for idx, (input, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            logits = self.model(input)

            loss = self.loss_func(logits, label)
            loss.backward()
            self.optimizer.step()

            if idx % 100 == 0:
                loss, current = loss.item(), idx * self.batch_size + len(input)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_loader.dataset):>5d}]")

    def test(self):
        self.model.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            for input, label in self.test_loader:
                logits = self.model(input)
                test_loss += F.cross_entropy(logits, label).item()
                correct += (logits.argmax(1) == label).type(torch.float).sum().item()

            test_loss /= len(self.test_loader)
            correct /= len(self.test_loader.dataset)
            print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f}\n")