# 🧠 MediScan AI - Brain Tumor Segmentation using Deep Learning

**MediScan AI** is a medical imaging web application that uses deep learning to segment brain tumors from MRI scans. Built using U-Net and PyTorch, this tool performs accurate tumor segmentation and generates detailed visual and PDF reports to support radiologists in diagnosis.

---

## 🔍 Problem Statement

Early detection of brain tumors can significantly improve patient outcomes, but manual diagnosis through MRI scans is time-consuming and prone to human error. This project aims to assist radiologists by providing an AI-powered tool for accurate and fast tumor segmentation and diagnosis reporting.

---

## 📌 Project Description

MediScan AI is a deep learning-based medical tool for brain MRI image segmentation. It enables users to upload MRI scans, processes them using a pre-trained U-Net model, and returns:

- A segmented image (highlighting tumor area)
- A confidence score
- An auto-generated medical report (JSON + PDF)

**Key Features:**
- Upload single or multiple images
- View original and segmented images side by side
- Download segmented image
- Generate PDF and JSON reports including pathology, volume, and comparative analysis
- Lightweight and runs locally using Streamlit

---

## 👥 Who Will Benefit?

- Radiologists and doctors for faster second-opinion analysis  
- Medical students and researchers in the healthcare AI domain  
- Rural clinics with limited radiology access  
- Hospitals dealing with large volumes of brain MRI scans  

---

## 💡 What Makes It Unique?

- Combines AI segmentation with real-time PDF report generation  
- Simple and accessible interface using Streamlit  
- Works offline – no need for cloud access or external APIs  
- Expandable to other medical segmentation tasks  

---

## 🧰 Technologies & Tools Used

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Deep Learning:** PyTorch, U-Net  
- **Image Processing:** MONAI, NumPy, Pillow  
- **PDF Generation:** FPDF  
- **Visualization:** Matplotlib  

---

## 🧪 Dataset Used

- **Name:** Brain MRI Images for Brain Tumor Detection  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)  
- **Classes:** Meningioma, Glioma, Pituitary Tumor, No Tumor  
- **Preprocessing:** Resize to 256x256, normalization, conversion to tensor  
