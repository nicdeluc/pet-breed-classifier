import torch
from torchvision import models
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import wandb
from utils_tune import get_pet_data, load_model, fine_tune
from functools import partial  # Import partial


def sweep_run(train_data, test_data):  # Modified to accept data as arguments
    """
    Function to perform a single run of the sweep
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with wandb.init(dir="."):

        config = wandb.config

        # Create DataLoaders
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        dataloaders = train_dataloader, test_dataloader

        num_classes = len(train_data.classes)

        # Build model dynamically
        if config.architecture == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif config.architecture == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        model = load_model(model, num_classes, dropout_p=config.dropout_p)

        loss_fn = nn.CrossEntropyLoss()

        # Build optimizers dynamically
        optim_head = optim.Adam(model.fc.parameters(), lr=config.lr_head)
        optim_full = optim.Adam(
            model.parameters(), lr=config.lr_full, weight_decay=config.weight_decay_full
        )

        # Fine tune the model
        fine_tune(
            model=model,
            dataloaders=dataloaders,
            loss_fn=loss_fn,
            optimizers=(optim_head, optim_full),
            epochs_head=config.epochs_head,
            epochs_full=config.epochs_full,
            device=device,
        )


def main():
    # Create sweep with command: wandb sweep config/sweep_config.yaml
    # Take sweep ID
    sweep_id = "u9axl3oo"

    print("Loading dataset...")
    train_data, test_data = get_pet_data()
    print("Dataset loaded successfully.")

    # Fill the sweep function with the data
    sweep_fn = partial(sweep_run, train_data=train_data, test_data=test_data)

    # Run the agent
    print(f"Starting W&B agent for sweep: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=sweep_fn,
        count=1,
        entity="nicdeluc-learning",
        project="pet-breed-classification",
    )


if __name__ == "__main__":
    main()
