# create_dummy_model.py
import torch
import torch.nn as nn
from utils_classification import SimpleCNN

model = SimpleCNN()
torch.save(model.state_dict(), "classification_model.pth")
print("Dummy model saved.")
