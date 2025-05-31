import torch
import torch.nn as nn
import cv2
import numpy as np

# SimpleCNN model matching saved model keys
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_classifier_model(model_path="classification_model.pth", device="cpu"):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_for_classification(image, target_size=(128, 128)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # (1, H, W)
    image = np.expand_dims(image, axis=0)  # (1, 1, H, W)
    return torch.tensor(image).to(torch.float32)
