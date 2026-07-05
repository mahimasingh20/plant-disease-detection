import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_plant_model.keras")

model = load_model()

# -------------------- Load Class Names --------------------
with open("class_names.json", "r") as file:
    class_names = json.load(file)

# -------------------- App Title --------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to identify the disease.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- Prediction --------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Image preprocessing
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result = class_names[predicted_index]

    # Separate Plant and Disease
    if "___" in result:
        plant, disease = result.split("___")
    else:
        plant = result
        disease = "Unknown"

    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ")

    # -------------------- Result --------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    st.write(f"**Plant Name:** {plant}")

    st.write(f"**Disease:** {disease}")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(confidence / 100)

    # Classification
    if "healthy" in disease.lower():

        st.success("🟢 Classification: Healthy Plant")

        st.subheader("Recommendation")

        st.write("""
- Continue regular watering.
- Provide sufficient sunlight.
- Use balanced fertilizers.
- Monitor leaves regularly.
- Maintain proper soil moisture.
        """)

    else:

        st.error("🔴 Classification: Diseased Plant")

        st.subheader("Recommendation")

        st.write("""
- Remove infected leaves immediately.
- Avoid overwatering.
- Improve air circulation around the plant.
- Apply a suitable fungicide or pesticide if required.
- Keep infected plants away from healthy plants.
        """)

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed by Mahima Singh | MobileNetV2 | TensorFlow | Streamlit")
