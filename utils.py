# utils.py
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import exposure
from scipy.ndimage import gaussian_filter

def load_image(path):
    if path.endswith(".dcm"):
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img = normalize_image(img)
        return img, dcm
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = img.astype(np.float32)
        img = normalize_image(img)
        return img, None

def normalize_image(img):
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)  # Z-score normalization
    img = np.clip(img, -3, 3)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # [0,1]
    return img

def preprocess_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = exposure.equalize_hist(img)
    return img
def load_dicom_image(path):
    """Load DICOM image and metadata"""
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)

    # Normalize image values
    img = z_score_normalization(img)

    # Apply histogram equalization (optional)
    img = exposure.equalize_hist(img)

    # Apply Gaussian filter to denoise
    img = gaussian_filter(img, sigma=1)

    metadata = {
        "PatientID": dicom.get("PatientID", "Unknown"),
        "Modality": dicom.get("Modality", "Unknown"),
        "StudyDate": dicom.get("StudyDate", "Unknown"),
        "Manufacturer": dicom.get("Manufacturer", "Unknown")
    }

    return img, metadata

def z_score_normalization(img):
    """Standardize image using z-score normalization"""
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std if std != 0 else img
