from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# this code is directly imported from the flower tutorial with no modifications
# creates a simple convolutional neural network for training an image classifier with 10 classes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs, learning_rate=0.01):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        for batch in trainloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_acc = correct / total
        # print(f'Epoch {epoch + 1}: train accuracy {epoch_acc}')


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / total
    loss /= len(testloader.dataset)
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
