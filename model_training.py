from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import Compose, LoadImage, AddChannelD, Resize, ScaleIntensity, ToTensor
from monai.data import Dataset, DataLoader
import torch

# Dummy dataset (replace with your real dataset paths)
train_images = ["path/to/image1.png", "path/to/image2.png"]
train_labels = ["path/to/mask1.png", "path/to/mask2.png"]

# Compose transforms using dictionary-based transforms with keys
train_transforms = Compose([
    LoadImage(image_only=True),
    AddChannelD(keys=["image", "label"]),
    Resize(keys=["image", "label"], spatial_size=(256, 256)),
    ScaleIntensity(keys=["image"]),
    ToTensor(keys=["image", "label"])
])

# Prepare data dictionary list
train_data = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

# Create Dataset and DataLoader
train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    dimensions=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# Loss and optimizer
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# Training loop
for epoch in range(10):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# Save model state_dict
torch.save(model.state_dict(), "models/unet.pth")
