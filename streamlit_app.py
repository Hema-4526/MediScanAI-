import io
import json
import torch
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
import streamlit as st
from save_model import UNet
import random
import os

# Streamlit Config
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="AI Medical Segmentation", layout="wide")
st.title("🩺MediScan AI")
st.markdown("#### AI-Powered Medical Image Segmentation Tool")
st.markdown("---")

# Image transform
transform = Compose([
    Resize((256, 256)),
    ScaleIntensity(),
    ToTensor()
])

# Load model
def load_model(model_path, device):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess image
def preprocess_image(uploaded_file, device):
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    image_np = np.array(image) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).to(device)
    return image, image_tensor

# Inference
def run_inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = torch.sigmoid(output).cpu().squeeze().numpy()
        if prob_map.ndim == 3:
            prob_map = prob_map[0]
        mask = (prob_map > 0.5).astype(np.uint8)
        confidence_score = float(np.mean(prob_map)) * 100
    return mask, confidence_score

# Visualization
def visualize_results(original_img, mask):
    fig, ax = plt.subplots()
    ax.imshow(original_img)
    ax.imshow(mask, alpha=0.4, cmap='Reds')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# Mock report
def generate_mock_report(image_name, confidence_score):
    pathologies = ["Glioblastoma", "Meningioma", "No Abnormality", "Ischemic Lesion", "Edema"]
    locations = ["Left temporal lobe", "Frontal cortex", "Occipital region", "Right parietal lobe", "Brainstem"]
    patient_id = f"P{random.randint(1000,9999)}"
    volume = round(random.uniform(1.5, 8.5), 2)
    selected_pathology = random.choice(pathologies)
    report = {
        "PatientID": patient_id,
        "ScanType": "MRI-Brain",
        "Findings": [{
            "Pathology": selected_pathology,
            "Confidence": round(confidence_score / 100, 2),
            "Location": random.choice(locations),
            "BoundingBox": [32, 48, 128, 180],
            "Volume_cm3": volume
        }],
        "ReportSummary": (
            "No abnormality" if selected_pathology == "No Abnormality"
            else f"{selected_pathology} with potential mass effect."
        ),
        "ComparativeAnalysis": (
            "No prior scans available." if random.random() < 0.5
            else "20% increase compared to previous scan."
        )
    }
    return report

# PDF report
def save_report_as_pdf(report, original_img, segmented_img_buf):
    def safe_str(s):
        return str(s).encode("latin-1", errors="replace").decode("latin-1")

    # Save original and segmented images temporarily
    original_img_path = "temp_original.png"
    segmented_img_path = "temp_segmented.png"
    original_img.save(original_img_path)
    with open(segmented_img_path, "wb") as f:
        f.write(segmented_img_buf.read())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Medical Scan Report")
    pdf.cell(200, 10, txt=safe_str("Medical Scan Report"), ln=True, align='C')
    pdf.ln(10)

    # Display images side-by-side
    pdf.cell(95, 10, txt="Original Image", ln=0, align='C')
    pdf.cell(95, 10, txt="Segmented Output", ln=1, align='C')

    y = pdf.get_y()
    pdf.image(original_img_path, x=10, y=y, w=90)
    pdf.image(segmented_img_path, x=110, y=y, w=90)
    pdf.ln(75)

    pdf.ln(5)
    for key, value in report.items():
        if isinstance(value, list):
            pdf.cell(200, 10, txt=safe_str(f"{key}:"), ln=True)
            for item in value:
                for k, v in item.items():
                    pdf.cell(200, 10, txt=safe_str(f"    {k}: {v}"), ln=True)
        else:
            pdf.cell(200, 10, txt=safe_str(f"{key}: {value}"), ln=True)

    # Convert to buffer
    pdf_output = pdf.output(dest='S').encode("latin-1")
    buffer = io.BytesIO(pdf_output)
    buffer.seek(0)

    # Cleanup
    os.remove(original_img_path)
    os.remove(segmented_img_path)

    return buffer


def create_side_by_side_image(original_img, mask):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").resize(original_img.size)
    overlay = Image.blend(original_img.convert("RGB"), mask_img, alpha=0.4)

    side_by_side = Image.new("RGB", (original_img.width * 2, original_img.height))
    side_by_side.paste(original_img, (0, 0))
    side_by_side.paste(overlay, (original_img.width, 0))

    buf = io.BytesIO()
    side_by_side.save(buf, format="PNG")
    buf.seek(0)
    return buf, side_by_side

# Main App
def main():
    with st.sidebar:
        uploaded_files = st.file_uploader("📁 Upload Images (jpg, png)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        st.markdown("---")
        st.markdown("👈 Upload one or more images to begin segmentation.")

    if uploaded_files:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model("models/unet.pth", device)

        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"## 🖼️ Image {idx + 1}: {uploaded_file.name}")
            with st.spinner("🧠 Running inference..."):
                try:
                    original_img, image_tensor = preprocess_image(uploaded_file, device)
                    mask, confidence_score = run_inference(model, image_tensor)
                    buf = visualize_results(original_img, mask)
                    side_by_side_buf, side_by_side_img = create_side_by_side_image(original_img, mask)
                    report = generate_mock_report(uploaded_file.name, confidence_score)
                    report_json = json.dumps(report, indent=2)
                    pdf_buf = save_report_as_pdf(report, original_img, buf)
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                    continue

            st.success(f"✅ Segmentation complete for {uploaded_file.name}")
            st.markdown(f"🔎 **Model Confidence Score:** `{confidence_score:.2f}%`")

            col1, col2 = st.columns(2)
            col1.image(original_img, caption="Original Image", use_container_width=True)
            col2.image(buf, caption="Segmented Output", use_container_width=True)

            st.image(side_by_side_img, caption="🖼️ Original + Segmented (Side-by-Side)", use_container_width=True)
            st.download_button("📥 Download Side-by-Side Image", side_by_side_buf, file_name=f"{uploaded_file.name}_side_by_side.png")
            st.download_button("📥 Download Segmented Mask Only", buf, file_name=f"{uploaded_file.name}_segmented_mask.png")


            with st.expander("📄 View Auto-Generated Report"):
                st.markdown("### 🧾 Patient Report")

                st.markdown(f"**🆔 Patient ID:** `{report['PatientID']}`")
                st.markdown(f"**🧪 Scan Type:** `{report['ScanType']}`")
                st.markdown("---")

                finding = report["Findings"][0]

                st.markdown("### 🔍 Findings Summary")
                st.table({
                    "Pathology": [finding["Pathology"]],
                    "Confidence": [f"{finding['Confidence']*100:.2f}%"],
                    "Location": [finding["Location"]],
                    "Volume (cm³)": [finding["Volume_cm3"]],
                    "Bounding Box": [str(finding["BoundingBox"])]
                })

                st.markdown("---")
                st.markdown(f"**📝 Report Summary:** {report['ReportSummary']}")
                st.markdown(f"**📊 Comparative Analysis:** {report['ComparativeAnalysis']}")

                st.download_button("⬇️ Download Report (JSON)", data=report_json, file_name=f"{uploaded_file.name}_report.json", mime="application/json")
                st.download_button("📄 Download Report (PDF)", data=pdf_buf, file_name=f"{uploaded_file.name}_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
