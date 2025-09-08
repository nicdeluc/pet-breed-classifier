import gradio as gr
from utils_app import load_model, prediction_probabilities
import torch
from torchvision import models

# Define variables
MODEL_PATH = "./assets/models/fine_tuned_resnet34.pth"
LABEL_TO_BREED = {
    0: "Abyssinian",
    1: "American Bulldog",
    2: "American Pit Bull Terrier",
    3: "Basset Hound",
    4: "Beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "Boxer",
    9: "British Shorthair",
    10: "Chihuahua",
    11: "Egyptian Mau",
    12: "English Cocker Spaniel",
    13: "English Setter",
    14: "German Shorthaired",
    15: "Great Pyrenees",
    16: "Havanese",
    17: "Japanese Chin",
    18: "Keeshond",
    19: "Leonberger",
    20: "Maine Coon",
    21: "Miniature Pinscher",
    22: "Newfoundland",
    23: "Persian",
    24: "Pomeranian",
    25: "Pug",
    26: "Ragdoll",
    27: "Russian Blue",
    28: "Saint Bernard",
    29: "Samoyed",
    30: "Scottish Terrier",
    31: "Shiba Inu",
    32: "Siamese",
    33: "Sphynx",
    34: "Staffordshire Bull Terrier",
    35: "Wheaten Terrier",
    36: "Yorkshire Terrier",
}
NUM_CLASSES = len(LABEL_TO_BREED)

## Load the model ##
print("Loading model...")
# Create the model structure
model = models.resnet34(weights=None)
model = load_model(model, num_classes=NUM_CLASSES, device="cpu")  # Modify the final layer
# Load the trained weights
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()
print("Model loaded successfully!")


def predict(image):
    """
    Takes an image and uses a fine-tuned ResNet-34 to predict the label and probability.
    """
    pred_probs = prediction_probabilities(image, model)
    # Squeeze the batch dimension and convert tensor to a simple list
    probs_list = pred_probs[0].tolist()

    # Create dictionary with the probability (confidence) for each pet breed
    confidences = {LABEL_TO_BREED[i]: probs_list[i] for i in range(NUM_CLASSES)}

    return confidences


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Pet Breed Classifier",
    description="This is a demo of a ResNet-34 model fine-tuned on 37 classes of pet breed images. Upload an image to see its prediction.",
    examples=[
        ["./assets/images/Abyssinian_9.jpg"],
        ["./assets/images/great_pyrenees_56.jpg"],
        ["./assets/images/Persian_85.jpg"],
        ["./assets/images/pomeranian_50.jpg"]
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
