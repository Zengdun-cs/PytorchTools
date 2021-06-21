import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision

def get_dataset(dataset='MNIST', transform=None, root='/home/zengdun/datasets/mnist/'):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=test_transform)
   
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=128,
                                              drop_last=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             drop_last=False, num_workers=2, shuffle=False) 
    return trainloader, testloader


class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
import time

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        

def get_dataset(dataset='MNIST', batch_size=128, root='/home/zengdun/datasets/mnist/'):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform)
   
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),drop_last=False, shuffle=False) 
    return trainloader, testloader

def evaluate(model, criterion, test_loader, cuda):

    model.eval()
    loss_sum = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted.eq(labels)).item()
            total += len(labels)
            loss_sum += loss.item()

    accuracy = correct / total
    print("Evaluate, Loss {:.4f}, accuracy: {:.4f}".format(loss_sum,accuracy))

if __name__ == "__main__":
    model = mlp().cuda()
    train_loader, test_loader = get_dataset()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        start_time = time.time()
        for step, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
    
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.detach().item()
        
        end_time = time.time()
        print("Epoch {}/{}, Loss: {:.4f}, Time cost: {:.2f}".format(epoch + 1, epochs, loss_sum, end_time-start_time))
    evaluate(model, criterion, test_loader, cuda=True)