import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nn_utils import *

class FrozenNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        conv1 = nn.Conv2d(3, 6, 5)
        fixed_indices = torch.tensor([0], dtype=torch.long)
        trainable_indices = torch.tensor([1,2,3,4,5], dtype=torch.long)
        self.conv1 = FrozenConv2d(trainable_indices, fixed_indices, conv1.weight.data.detach().clone(),
                                  conv1.bias.data.detach().clone())
        
        self.pool = nn.MaxPool2d(2, 2)
        
        conv2 = nn.Conv2d(6, 16, 5)
        fixed_indices2 = torch.tensor([0], dtype=torch.long)
        trainable_indices2 = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=torch.long)
        self.conv2 = FrozenConv2d(trainable_indices2, fixed_indices2, conv2.weight.data.detach().clone(),
                                  conv2.bias.data.detach().clone())
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1.fixed_indices = torch.tensor([0], dtype=torch.long)
        self.conv1.trainable_indices = torch.tensor([1,2], dtype=torch.long)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2.fixed_indices = torch.tensor([0,1,2], dtype=torch.long)
        self.conv2.trainable_indices = torch.tensor([3,4,5], dtype=torch.long)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Initialize dataset and dataloaders
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize network
    # net = Net()
    net = FrozenNet()
    
    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()