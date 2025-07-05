# Streamlit app: YOLOv8 + Caption Generator + Feedback Tracker

import streamlit as st
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from huggingface_hub import login
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter

# -----------------------------
# ✅ Authenticate Hugging Face
# -----------------------------
hf_token = st.secrets.get("hf_token") or os.getenv("HF_TOKEN")

if hf_token:
    login(hf_token)
else:
    st.warning("⚠️ Hugging Face token not found. You might hit download rate limits.")

# -----------------------------
# ✅ Load Models
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8 from Ultralytics
yolo_model = YOLO("yolov8n.pt")

# ViT-GPT2 from Hugging Face (captioning)
model_id = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# -----------------------------
# ✅ Helper Functions
# -----------------------------
@st.cache_data
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image

def detect_objects(image, image_id):
    temp_path = f"temp_{image_id}.jpg"
    image.save(temp_path)
    results = yolo_model(temp_path)
    labels = []
    draw = ImageDraw.Draw(image)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            labels.append(label)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            draw.rectangle(xyxy.tolist(), outline="red", width=3)
            draw.text((xyxy[0], xyxy[1] - 10), label, fill="red")
    os.remove(temp_path)
    return image, labels

def generate_captions(image, labels):
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(caption_model.device)
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=1)
    base_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    cap1 = base_caption
    cap2 = f"{base_caption}. Objects: {', '.join(labels[:2])}." if labels else cap1
    cap3 = f"{base_caption}. This image includes: {', '.join(labels)}." if labels else cap1
    return cap1, cap2, cap3

def save_feedback(image_id, caption_level):
    path = "feedback.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["image_id", "caption_level"])
    df = pd.concat([df, pd.DataFrame([[image_id, caption_level]], columns=["image_id", "caption_level"])]).reset_index(drop=True)
    df.to_csv(path, index=False)

def load_feedback_stats():
    path = "feedback.csv"
    if not os.path.exists(path):
        return Counter()
    df = pd.read_csv(path)
    return Counter(df["caption_level"])

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("🧠 Object Detection & Image Caption Generator ")

num_files = st.number_input("How many images do you want to upload?", min_value=1, max_value=10, step=1)
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != num_files:
        st.warning(f"You selected {num_files}, but uploaded {len(uploaded_files)} files.")
    else:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            image = load_image(uploaded_file)
            st.subheader(f"🖼️ Image {i}")

            image_with_boxes, labels = detect_objects(image.copy(), i)
            st.image(image_with_boxes, caption=f"Detected Objects: {', '.join(labels)}", use_container_width=True)


            cap1, cap2, cap3 = generate_captions(image, labels)
            st.markdown("**Choose the best caption:**")
            choice = st.radio(f"Image {i} Captions:", [cap1, cap2, cap3], key=f"radio_{i}")
            st.markdown(f"<small>Relative:</small> {cap1}", unsafe_allow_html=True)
            st.markdown(f"<small>More Relative:</small> {cap2}", unsafe_allow_html=True)
            st.markdown(f"<small>Mostly Relative:</small> {cap3}", unsafe_allow_html=True)

            if st.button(f"Submit Feedback for Image {i}"):
                if choice == cap1:
                    caption_level = "Related"
                elif choice == cap2:
                    caption_level = "More Related"
                else:
                    caption_level = "Mostly Related"
                save_feedback(i, caption_level)
                st.success("✅ Feedback submitted!")

        # -----------------------------
        # ✅ Pie Chart
        # -----------------------------
        st.markdown("---")
        st.subheader("📊 Caption Satisfaction Statistics")
        stats = load_feedback_stats()
        if stats:
            labels = list(stats.keys())
            sizes = list(stats.values())
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=80)
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No feedback yet.")
