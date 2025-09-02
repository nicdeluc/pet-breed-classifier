import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from utils import get_pet_data, load_model, track_experiment, plot_history, save_model

def main():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    ### Data ###

    # Get the data
    train_data, test_data = get_pet_data()

    # Construct the dataloaders 
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    dataloaders = train_dataloader, test_dataloader

    class_names = train_data.classes
    num_classes = len(class_names)

    ### Model ###

    # Set up the model
    resnet34 =  models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    resnet34 = load_model(resnet34, num_classes=num_classes, dropout_p=0.5)

    ### Training ###

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizers for training the model
    optim_head = optim.Adam(resnet34.fc.parameters(), lr=0.002)
    optim_full = optim.Adam(resnet34.parameters(), lr=0.00001, weight_decay=0.01)
    optimizers = optim_head, optim_full

    # Define the parameters to train the model

    config = {
        'lr_head': 0.002,
        'lr_full': 0.00001,
        'dataset': 'OxfordIIITPet',
        'weight_decay_full': 0.01,
        'epochs_head': 10,
        'epochs_full': 30
    }

    # Train the model and track the experiment with W&B
    history= track_experiment(resnet34, dataloaders, loss_fn, optimizers, config, device)
    
    # Plot the result
    plot_history(history)
    
    # Save the model
    save_model(resnet34)
    
if __name__ == '__main__':
    main()