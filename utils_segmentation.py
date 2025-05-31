import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor
import numpy as np

def get_unet_model():
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model

def preprocess_for_unet(image):
    # Ensure image has channel dimension (C, H, W)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    transforms = Compose([
        Resize((128, 128)),
        ScaleIntensity(),
        ToTensor()
    ])
    tensor = transforms(image)
    return tensor.unsqueeze(0).float()  # batch dim added

def segment_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        mask = (pred > 0.5).float()
    return mask.cpu().squeeze().numpy()
