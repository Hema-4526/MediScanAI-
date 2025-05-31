import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils_classification import SimpleCNN, preprocess_for_classification

# Dummy dataset class - replace with your real dataset loader
class MedicalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Simple CNN model (replace with your architecture if you want)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*64*64, 2)  # Assuming input 128x128 resized images

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Replace these with your actual training images and labels as tensors
    train_images = torch.randn(100, 1, 128, 128)  # Example dummy data
    train_labels = torch.randint(0, 2, (100,))    # 0 or 1 labels

    transform = transforms.Normalize([0.5], [0.5])

    dataset = MedicalDataset(train_images, train_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

    # Save trained model weights
    torch.save(model.state_dict(), "classification_model.pth")
    print("Training done and model saved as classification_model.pth")

if __name__ == "__main__":
    train()
