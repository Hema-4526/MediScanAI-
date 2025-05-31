import cv2 
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dicom_image  # Replaces load_image
from utils_segmentation import get_unet_model, preprocess_for_unet
from utils_preprocessing import preprocess_image
import os

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the U-Net model and set to evaluation mode
model = get_unet_model().to(device)
model.eval()

# Load DICOM image and metadata
img_path = "uploads/sample_mri.dcm"
img, metadata = load_dicom_image(img_path)

print("Metadata:", metadata)
print("Image shape before processing:", img.shape)  # e.g. (96, 512, 512) for 3D volume

# Parameters
target_size = (128, 128)  # Resize size for model input
threshold = 0.5           # Probability threshold for mask binarization

# Create output directory for segmentation masks
output_dir = "segmentation_masks"
os.makedirs(output_dir, exist_ok=True)

# Handle 2D or 3D images uniformly by adding slice dimension if needed
if img.ndim == 3:
    num_slices = img.shape[0]
else:
    num_slices = 1
    img = np.expand_dims(img, axis=0)

# Process each slice
for slice_index in range(num_slices):
    print(f"Processing slice {slice_index + 1}/{num_slices}...")

    slice_img = img[slice_index, :, :]

    if slice_index == 0:
        print("Shape of one slice:", slice_img.shape)

    # Preprocess and resize slice for model input
    slice_preprocessed = preprocess_image(slice_img)
    slice_resized = cv2.resize(slice_preprocessed, target_size)

    # Prepare input tensor for U-Net
    input_tensor = preprocess_for_unet(slice_resized).to(device)

    # Predict segmentation mask
    with torch.no_grad():
        output = model(input_tensor)
        mask_prob = torch.sigmoid(output).cpu().numpy()[0, 0]

    # Threshold probability mask to binary mask
    mask_binary = (mask_prob > threshold).astype(np.uint8) * 255

    # Resize mask back to original slice size
    mask_resized = cv2.resize(mask_binary, (slice_img.shape[1], slice_img.shape[0]))

    # Save mask for first 5 slices only
    if slice_index < 5:
        mask_path = os.path.join(output_dir, f"mask_slice_{slice_index}.png")
        cv2.imwrite(mask_path, mask_resized)
        print(f"Saved {mask_path}")

    # Display the first slice and its mask
    if slice_index == 0:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Slice {slice_index}")
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Segmentation Mask Overlay")
        plt.imshow(slice_img, cmap='gray')
        plt.imshow(mask_resized, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()
