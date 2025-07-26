from torchvision.datasets import OxfordIIITPet
from torchvision import transforms


def get_data(dataset):
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
    return train_data, test_data