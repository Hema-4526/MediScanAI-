import torch
from utils_classification import load_classifier_model, preprocess_for_classification
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_classifier_model("classification_model.pth", device)

# Load your test images and true labels here
# For demo, replace with actual test set loader
test_images = ["test_img1.png", "test_img2.png"]  # Example
true_labels = [0, 1]  # Replace with actual labels: 0=No Tumor,1= Tumor

preds = []
for img_path in test_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    input_tensor = preprocess_for_classification(img).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        preds.append(pred)

acc = accuracy_score(true_labels, preds)
cm = confusion_matrix(true_labels, preds)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
