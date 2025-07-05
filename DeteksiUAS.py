import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Setup halaman
st.set_page_config(page_title="Deteksi Objek YOLOv8", layout="centered")
st.title("🎯 Deteksi Objek Otomatis dengan YOLOv8")

# Pilih model
model_choice = st.selectbox(
    "Pilih Model Deteksi:",
    ["YOLOv8n (default)", "Model Custom (best.pt)"]
)
model_path = "yolov8n.pt" if model_choice == "YOLOv8n (default)" else "best.pt"

# Pilih input
input_type = st.radio("Pilih Jenis Input:", ["Upload Gambar", "Upload Video"])

# Fungsi deteksi gambar
def detect_image(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    for r in results:
        return r.plot()

# ------------------------------
# 📤 Upload Gambar
# ------------------------------
if input_type == "Upload Gambar":
    st.subheader("🖼️ Upload Gambar Anda")
    uploaded_image = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Simpan sementara gambar
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_image.write(uploaded_image.read())
        st.image(temp_image.name, caption="Gambar Asli", use_column_width=True)

        # Deteksi otomatis
        with st.spinner("🔍 Mendeteksi objek..."):
            result_img = detect_image(temp_image.name, model_path)
            st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

# ------------------------------
# 📤 Upload Video
# ------------------------------
elif input_type == "Upload Video":
    st.subheader("🎞️ Upload Video Anda")
    uploaded_video = st.file_uploader("Unggah video (mp4/avi)", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Simpan sementara video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name)

        with st.spinner("🕵️ Mendeteksi objek dalam video..."):
            model = YOLO(model_path)
            results = model(temp_video.name, save=True)

        # Cari hasil video output
        detect_dir = "runs/detect"
        if os.path.exists(detect_dir):
            latest_folder = sorted(os.listdir(detect_dir))[-1]
            output_files = os.listdir(os.path.join(detect_dir, latest_folder))

            video_out = None
            for file in output_files:
                if file.endswith((".mp4", ".avi", ".mov")):
                    video_out = os.path.join(detect_dir, latest_folder, file)
                    break

            if video_out and os.path.exists(video_out):
                st.success("✅ Deteksi selesai!")
                st.video(video_out)
            else:
                st.warning("⚠️ Video hasil tidak ditemukan.")
