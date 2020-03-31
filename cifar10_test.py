from backbone import get_efficientnet_params, EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


CLASS_LABELS = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_dataloaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    return trainloader, testloader


def main():
    train_loader, test_loader = get_dataloaders()

    b0_params = get_efficientnet_params("efficientnet-b0")
    print(b0_params)
    net = EfficientNet(b0_params, len(CLASS_LABELS))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(2):
        running_loss = 0.1
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(
                    f"Epoch: {epoch}, step: {i}, loss: {running_loss/200:.6f}"
                )
            running_loss = 0.0

    print("Finished training ")


if __name__ == "__main__":
    main()
