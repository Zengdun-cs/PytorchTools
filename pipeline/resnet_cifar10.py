import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
import time
import torch.nn.functional as F
from ..model.resnet import ResNet18

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
    print("Evaluate, Loss {:.4f}, accuracy: {:.4f}".format(loss_sum, accuracy))


root='/home/zengdun/datasets/cifar10/'

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root=root,
                                        train=True,
                                        download=True,
                                        transform=train_transform)

testset = torchvision.datasets.CIFAR10(root=root,
                                        train=False,
                                        download=True,
                                        transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=128,
                                            drop_last=True,
                                            num_workers=2)
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=len(testset),
                                            drop_last=False,
                                            shuffle=False)

model = ResNet18().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epochs = 10

for epoch in range(epochs):
    loss_sum = 0.0
    start_time = time.time()
    for step, (data, target) in enumerate(trainloader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach().item()

    end_time = time.time()

    print("Epoch {}/{}, Loss: {:.4f}, Time cost: {:.2f}".format(
        epoch + 1, epochs, loss_sum, end_time - start_time))
evaluate(model, criterion, testloader, cuda=True)