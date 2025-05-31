import pydicom
import cv2
import numpy as np

def load_image(path):
    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        # Normalize pixel values to 0-255 uint8
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        metadata = {
            "PatientID": getattr(ds, "PatientID", "Unknown"),
            "StudyDate": getattr(ds, "StudyDate", "Unknown"),
            "Modality": getattr(ds, "Modality", "Unknown"),
            "PixelSpacing": getattr(ds, "PixelSpacing", [1.0, 1.0])
        }
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        metadata = {
            "PatientID": "Unknown",
            "StudyDate": "Unknown",
            "Modality": "Unknown",
            "PixelSpacing": [1.0, 1.0]
        }
    return img, metadata

def preprocess_image(img):
    print("Image shape before processing:", img.shape)

    # Normalize if not uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Handle 3D volume: pick middle slice
    if len(img.shape) == 3:
        img = img[img.shape[0] // 2]

    # Squeeze if shape is (H, W, 1)
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)

    # Convert RGB to grayscale if 3 channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (Histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # Gaussian Blur for noise reduction
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    # Z-score normalization (float32)
    mean = np.mean(img_blur)
    std = np.std(img_blur)
    img_norm = (img_blur - mean) / (std + 1e-8)

    return img_norm.astype(np.float32)

if __name__ == "__main__":
    # Test both files
    test_paths = [
        "D:/programming/AI_Medical_Imaging/uploads/sample_mri.dcm",
        "D:/programming/AI_Medical_Imaging/uploads/sample_mri.jpg"
    ]
    
    for test_path in test_paths:
        print(f"\nTesting file: {test_path}")
        img, metadata = load_image(test_path)
        print("Metadata:", metadata)
        if img is not None:
            processed_img = preprocess_image(img)
            print("Processed image shape:", processed_img.shape)
            print("Processed image stats - mean:", processed_img.mean(), "std:", processed_img.std())
        else:
            print("Failed to load image.")
