import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# load data
train_dataset = datasets.CIFAR10(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X**2]
    # STD = sqrt(VAR)
    CHANNELS_SUM, CHANNELS_SQUARED_SUM, NUM_BATCHES = 0, 0, 0

    for data, target in loader:
        CHANNELS_SUM += torch.mean(data, dim=[0, 1, 2])
        CHANNELS_SQUARED_SUM += torch.mean(data ** 2, dim=[0, 2, 3])
        NUM_BATCHES += 1

    mean = CHANNELS_SUM / NUM_BATCHES
    std = (CHANNELS_SQUARED_SUM / NUM_BATCHES - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)
# As the CIFAR10 dataset has RGB data it will return 3 elements in an array
# If you use the MNIST dataset then their will be a single element
print(f"Mean : {mean}")
print(f"STD : {std}")
