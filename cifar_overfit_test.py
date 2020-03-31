from backbone import get_efficientnet_params, EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


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
BATCH_SIZE = 32


def get_dataloader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )

    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    return dataloader


def main():
    dataloader = get_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    b0_params = get_efficientnet_params("efficientnet-b0")
    print(b0_params)
    net = EfficientNet(b0_params, len(CLASS_LABELS))
    net.to(device)
    summary(net, input_size=(3, 24, 24))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                if i == 0:
                    continue
                print(
                    f"Epoch: {epoch+1}, step: {i:04d}, loss: {running_loss/100:.4f}"
                )
                running_loss = 0.0

    print("Finished training ")
    torch.save(net.state_dict(), "./cifar_net.pth")

    # Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            outputs = net(images).to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Overfit accuracy: {1e2*correct/total:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
