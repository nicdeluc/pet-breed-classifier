import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_pet_data():
    """
    Downloads the OxfordIIITPet data set (both train and test) 
    applying the desired transformations to the images
    Returns:
        data: A tuple containing:
            - train_data: The OxfordIIITPet train set.
            - test_data: The OxfordIIITPet test set.
    """
    train_data = OxfordIIITPet(
        root="data",
        split="trainval",
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = OxfordIIITPet(
        root="data",
        split="test",
        transform=transforms.ToTensor(),
        download=True
    )
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    return train_data, test_data

def visualize_labeled_data(data, size):
    """
    Visualize a sample of labeled data
    """
    figure = plt.figure(figsize=(2*size, 2*size))
    for i in range(1, size**2 + 1):
        sample_idx = torch.randint(len(data))
        image, label = data[sample_idx]
        figure.subplot(size, size, i)
        plt.title(label[sample_idx])
        plt.axis("off")
        plt.imshow(image)
    plt.show()