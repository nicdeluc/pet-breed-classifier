import torch
from torch import nn
from torchvision import transforms


def load_model(model, num_classes, dropout_p=0.5, device="cuda"):
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
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_ftrs, num_classes))

    # Move model to device
    model.to(device)

    return model


def prediction_probabilities(image, model, device="cuda"):
    """ """
    pred_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = pred_transform(image)
    batch_tensor = image_tensor.unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)
        pred_probs = torch.nn.functional.softmax(logits, dim=1)

    return pred_probs
