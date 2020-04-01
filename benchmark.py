import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from backbone import EfficientNet


TRAIN_DIR = "data/imagenette-320/train"
N_CLASSES = 10
FREQ = 50


def make_data_loader():
    composed = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DIR, transform=composed
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32, num_workers=1, shuffle=True
    )
    return dataloader


def make_model():
    return EfficientNet.from_name("efficientnet-b0", N_CLASSES)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    dataloader = make_data_loader()

    net = make_model()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=2, verbose=True
    )

    for epoch in range(20):
        epoch_loss = 0.0
        running_loss = 0.0
        for batch_id, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if batch_id % FREQ == 0 and batch_id != 0:
                print(
                    f"Epoch: {epoch+1}, step: {batch_id:03d}, loss: {running_loss/FREQ:.4f}"
                )
                running_loss = 0.0

        print(f"End of Epoch {epoch+1}, loss: {epoch_loss/batch_id:.2f}")
        lr_scheduler.step(epoch_loss / batch_id)


if __name__ == "__main__":
    main()
