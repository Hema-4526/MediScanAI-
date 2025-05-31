import torch

# Load the model checkpoint
checkpoint = torch.load("models/unet.pth", map_location='cpu')

# Check if it's a full checkpoint with 'state_dict'
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # it's likely just the state_dict

# Print all layer names (keys)
print("Keys in the state_dict:")
for key in state_dict.keys():
    print(key)
