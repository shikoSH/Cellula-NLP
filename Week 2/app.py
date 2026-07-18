import streamlit as st
from PIL import Image
from imagecaption import generate_caption
from database import add_entry, load_db
# from text_classifier import classify_text  (your model, e.g. LoRA-tuned DistilBERT)

st.title("Toxic Content Classifier")

mode = st.radio("Input type", ["Text", "Image"])

if mode == "Text":
    text = st.text_area("Enter text")
    if st.button("Classify") and text:
        result = classify_text(text)   # your classifier call
        add_entry(text, result)
        st.write(f"Classification: {result}")

else:
    img_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if img_file and st.button("Generate caption & classify"):
        image = Image.open(img_file)
        caption = generate_caption(image)
        result = classify_text(caption)
        add_entry(caption, result)
        st.write(f"Caption: {caption}")
        st.write(f"Classification: {result}")

if st.button("View database"):
    st.dataframe(load_db())