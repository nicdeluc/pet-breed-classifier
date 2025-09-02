import torch
from torch import nn
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
import yaml


def get_pet_data():
    """
    Downloads the OxfordIIITPet data set (both train and test) 
    applying the desired transformations to the images
    Returns:
        data: A tuple containing:
            - train_data: The OxfordIIITPet train set.
            - test_data: The OxfordIIITPet test set.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalization of ImageSet (necessary if we use a model pre-trained on ImageSet, such as ResNet-18)
                            std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    train_data = OxfordIIITPet(
        root="data",
        split="trainval",
        transform=train_transform,
        download=True
    )
    test_data = OxfordIIITPet(
        root="data",
        split="test",
        transform=test_transform,
        download=True
    )
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    return train_data, test_data

def visualize_labeled_data(data, size):
    """
    Visualize a sample of labeled data
    """
    plt.figure(figsize=(2*size, 2*size))
    for i in range(1, size**2 + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        image, label = data[sample_idx]
        plt.subplot(size, size, i)
        plt.title(label[sample_idx])
        plt.axis("off")
        plt.imshow(image)
    plt.show()
    
def visualize_preproc_labeled_img(dataloader, label_to_breed):
    # Display some pre-processed image and label
    features, labels = next(iter(dataloader))  # Get next batch of the iterable
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    image = features[0]
    label = labels[0]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(label_to_breed[label.item()])
    plt.axis("off")
    
def load_model(model, num_classes, dropout_p=0.5, device='cuda'):
    """
    Load a pre-trained model and set it up for training:
    replace the last fully-connected layer and freeze
    the rest of the model.
    """
    # Freeze all the layers in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    # Create a new fully-connected layer for our new classes
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_ftrs, num_classes)
    ) 

    # Move model to device
    model.to(device)
    
    return model

def train_model(model, dataloaders, loss_fn, optimizer, epochs=10, device='cuda'):
    """
    """ 
    # Create a dictionary to store training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    train_dataloader, test_dataloader = dataloaders

    # Loop through epochs
    for epoch in range(epochs):
        ### Training Phase ###
        model.train()
        
        train_loss = 0.0
        for X, y in tqdm(train_dataloader, desc=f'Epoch {epoch+1} Training', leave=False):
            # Move data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

        # Calculate average training loss for the epoch
        train_loss /= len(train_dataloader)

        ### Validation Phase ###
        model.eval() # Set model to evaluation mode
        
        val_loss, val_acc = 0.0, 0.0
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                
                # Forward pass
                val_pred = model(X)
                
                # Calculate loss and accuracy
                val_loss += loss_fn(val_pred, y).item()
                val_acc += (val_pred.argmax(dim=1) == y).sum().item()

        # Calculate metrics over the whole validation set
        val_loss /= len(test_dataloader)
        val_acc /= len(test_dataloader.dataset)

        # Print progress
        print(f"Epoch: {epoch+1} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Val acc: {val_acc:.4f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
    return history

def fine_tune(model, dataloaders, loss_fn, optimizers, epochs_head=5, epochs_full=5, device='cuda'):
    """
    """
    optimizer_head, optimizer_full = optimizers
    
    # Train the head of the model
    print('Training model\'s head...')
    history_head = train_model(model, dataloaders, loss_fn, optimizer_head, epochs=epochs_head, device=device)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    print('Training full model...')
    # Train the full model
    history_full = train_model(model, dataloaders, loss_fn, optimizer_full, epochs=epochs_full, device=device)
    print('Finished Training')
    
    # Join the histories
    history = {'train_loss': history_head['train_loss'] + history_full['train_loss'], 
               'val_loss': history_head['val_loss'] + history_full['val_loss'], 
               'val_acc': history_head['val_acc'] + history_full['val_acc']}
    
    return history

def track_experiment(model, dataloaders, loss_fn, optimizers, config, device="cuda"):
    wandb.init(
        entity = 'nicdeluc-learning',
        project = 'pet-breed-classification',
        config = config,
        dir = '.'
    )

    # Fine-tune the model
    history = fine_tune(model, dataloaders, loss_fn, optimizers, epochs_head=10, epochs_full=20, device=device)

    # End experiment tracking
    wandb.finish()
    
    return history

# Funtion to plot training and test losses and accuracy
def plot_history(history):
    # Extract data from the history dictionary
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    
    # Create an x-axis for epochs
    epochs = range(1, len(train_loss) + 1)
    
    # Create a figure with two subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: training and validation loss
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, val_acc, 'g-o', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
def save_model(model):
    PATH = f'../tuned_models/fine_tuned_{model}.pth'
    torch.save(model.state_dict(), PATH)
    
def prediction_probabilities(image, model, device='cuda'):
    """
    """
    pred_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = pred_transform(image)
    batch_tensor = image_tensor.unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)
        pred_probs = torch.nn.functional.softmax(logits, dim=1)
        
    return pred_probs

def load_config(config_path):
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            # Use safe_load for security
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file: {e}")
        return None