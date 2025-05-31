import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define MONAI's UNet (same as in training)
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Load model weights
model_path = 'models/unet.pth'
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load and preprocess test image
test_image_path = 'images/000009.png'
image = Image.open(test_image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

input_tensor = transform(image).unsqueeze(0).to(device)

# Predict mask
with torch.no_grad():
    output = model(input_tensor)
    predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()

# Display original + predicted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap='gray')

plt.tight_layout()
plt.show()

# Save predicted mask as image
predicted_mask_img = Image.fromarray((predicted_mask * 255).astype('uint8'))
os.makedirs('output', exist_ok=True)
output_path = 'output/predicted_mask.png'
predicted_mask_img.save(output_path)

print(f"✅ Predicted mask saved to: {output_path}")
