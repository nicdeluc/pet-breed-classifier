import gradio as gr
from utils import load_model, predict_label
import torch
from torchvision import models

# Define variables
MODEL_PATH = '../tuned_models/fine_tuned_resnet34.pth'
NUM_CLASSES = 10

## Load the model ##
print("Loading model...")
# Create the model structure
model = models.resnet34(weights=None)
model = load_model(model, num_classes=NUM_CLASSES) # Modify the final layer
# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda', weights_only=True))
model.eval()
print("Model loaded successfully!")

def predict(image):
    """
    Takes an image and uses a fine-tuned ResNet-34 to predict the label and probability.
    """
    label, prob = predict_label(image, model)
    
    return {label: prob}

demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type="pil", label="Upload an Image"),
    outputs = gr.Label(num_top_classes=1, label="Prediction"),
    description="This is a demo of a fine-tuned ResNet34 model. Upload an image to see its prediction.",
    examples=[["./examples/cat.jpg"], ["./examples/dog.jpg"]]
    )

if __name__ == "__main__":
    demo.launch()
