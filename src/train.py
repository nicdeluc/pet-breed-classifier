import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from utils import get_pet_data, load_model, track_experiment, plot_history

def main():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ### Data ###

    # Get the data
    train_data, test_data = get_pet_data()

    # Construct the dataloaders 
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    dataloaders = train_dataloader, test_dataloader

    class_names = train_data.classes
    num_classes = len(class_names)
    label_to_breed = {i:name for i, name in enumerate(class_names)}  # Map between the labels and the names of the pet breeds

    ### Models ###

    # Set up ResNet-18
    resnet18 =  models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18 = load_model(resnet18, num_classes=num_classes, dropout_p=0.5)

    # Set up ResNet-34
    resnet34 =  models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    resnet34 = load_model(resnet34, num_classes=num_classes, dropout_p=0.5)

    ### Training ###

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizers for training ResNet-18
    optim_18_head = optim.Adam(resnet18.fc.parameters(), lr=0.002)
    optim_18_full = optim.Adam(resnet18.parameters(), lr=0.00001, weight_decay=0.01)
    optimizers_18 = optim_18_head, optim_18_full

    # Optimizers for training ResNet-34
    optim_34_head = optim.Adam(resnet34.fc.parameters(), lr=0.002)
    optim_34_full = optim.Adam(resnet34.parameters(), lr=0.00001, weight_decay=0.01)
    optimizers_34 = optim_34_head, optim_34_full

    # Define the parameters to train the models
    config_18 = {
            'lr_head': 0.002,
            'lr_full': 0.00001,
            'architecture': 'ResNet-18',
            'dataset': 'OxfordIIITPet',
            'weight_decay_full': 0.01,
            'epochs_head': 10,
            'epochs_full': 30
        }

    config_34 = {
        'lr_head': 0.002,
        'lr_full': 0.00001,
        'architecture': 'ResNet-34',
        'dataset': 'OxfordIIITPet',
        'weight_decay_full': 0.01,
        'epochs_head': 10,
        'epochs_full': 30
    }

    # Train the models and track the experiments with W&B
    history_resnet_18 = track_experiment(resnet18, dataloaders, loss_fn, optimizers_18, config_18)
    history_resnet_34 = track_experiment(resnet34, dataloaders, loss_fn, optimizers_34, config_34)
    
    # Plot the results
    plot_history(history_resnet_18)
    plot_history(history_resnet_34)
    
    # Save the models
    