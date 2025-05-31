import torch
from utils_classification import load_classifier_model, preprocess_for_classification
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load an example image (replace with your actual image loading)
image = np.random.rand(128, 128).astype(np.float32)  # Dummy image for test
print("Image shape:", image.shape)

model = load_classifier_model("classification_model.pth", device)

input_tensor = preprocess_for_classification(image).to(device)
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    classes = ["No Tumor", "Tumor"]
    print("Prediction:", classes[predicted.item()])
