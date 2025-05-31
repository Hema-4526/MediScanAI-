# test_preprocessing.py
from utils import load_image, preprocess_image
import matplotlib.pyplot as plt

image_path = 'uploads/sample_mri.jpg'  # Or .dcm file

img, meta = load_image(image_path)
preprocessed = preprocess_image(img)

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Preprocessed")
plt.imshow(preprocessed, cmap='gray')

plt.show()
