import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

# config page
st.set_page_config(
    page_title="ASL",
    page_icon="none",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Nhận diện chữ bằng ngôn ngữ kí hiệu")

# -----------------------
# load model
# -----------------------
@st.cache_resource
def load_model(model_url):
    model = keras.models.load_model(model_url)
    return model

model = load_model("asl_model.h5")

# -----------------------
# tao danh sach lop (class)
# -----------------------

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'
]
IMG_SIZE = 64

# -------------------------
# ham ho tro (xu ly hinh anh input)
# -------------------------
def preprocess_image(pil_img, input_size=(64, 64)):
    pil_img = pil_img.convert("RGB")  # Convert về RGB (tránh lỗi với hình RGBA)
    img = pil_img.resize(
        input_size
    )  # Resize hình cho phù hợp với input size của mô hình
    img_array = image.img_to_array(
        img
    )  # Chuyển từ kiểu dữ liệu hình sang kiểu numpy array

    img_array = np.expand_dims(img_array, axis=0)  # Thêm n=1 để batch_size=1
    test_datagen = image.ImageDataGenerator(  # Bắt buộc áp dụng các phương pháp tiền xử lý như tập train
        samplewise_center=True, samplewise_std_normalization=True
    )
    img_generator = test_datagen.flow(
        img_array, batch_size=1
    )  # Thay vì sử dụng `flow_from_directory` thì chỉ sử dụng `flow`
    return img_generator

# -----------------------
# tao giao dien
# -----------------------
# chon kieu test
input_type = st.selectbox("Choose a type: ", ("Upload image", "Use camera"), index=0)

# kiem tra kieu chon
if input_type == "Use_camera":
    st.warning("This feature is not done!")

# khung input hinh
uploaded_file = st.file_uploader(
    "Upload an image of a hand sign", type=['jpg', 'jpeg', 'png']
)

# not found hinh anh upload
if input_type == "Upload image" and uploaded_file is None:
    st.error("The image is not found!")
elif input_type == "Upload image"and uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

if st.button("Predict"):
    # loading
    with st.spinner("Predicting..."):
        img_input = preprocess_image(img)

        # du doan hinh anh
    predictions = model.predict(img_input)
    prediction_idx = np.argmax(predictions)
    predicted_label = class_names[prediction_idx]
    confidence = np.max(predictions)
    st.write(f"**Prediction:** {predicted_label} with {confidence*100:.2f}% confidence.")