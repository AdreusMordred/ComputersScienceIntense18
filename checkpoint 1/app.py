import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
from pathlib import Path

# Config trang
st.set_page_config(
    page_title="Dogs and Cats Classification",
    page_icon="none",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Nhận diện chó và mèo bằng hình ảnh")

# Load model — dùng @st.cache_resource để
# không load lại mỗi lần người dùng tương tác

@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# Thử load format mới (.keras) trước, fallback về .h5
# Use the script directory so model paths work regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent

try:
    model_path = BASE_DIR / "d&c_classification_model_best.keras"
    model = load_model(str(model_path))  # model tốt nhất từ ModelCheckpoint
    st.sidebar.success(f"Loaded: {model_path.name}")
except Exception:
    try:
        model_path = BASE_DIR / "d&c_classification_model.keras"
        model = load_model(str(model_path))
        st.sidebar.success(f"Loaded: {model_path.name}")
    except Exception as e:
        st.error(f"Không tìm thấy model: {e}")
        st.stop()

# Danh sách class — thứ tự phải khớp với
# class_indices lúc train

class_names = ["Cat", "Dog"]
IMG_SIZE = 224  # Phải khớp với IMG_SIZE lúc train MobileNetV2

# Hàm tiền xử lý ảnh — PHẢI giống hệt lúc train

def preprocess_image(pil_img):
    """
    Chuẩn bị ảnh đầu vào cho model MobileNetV2.
    Pipeline: RGB → Resize → numpy array → preprocess_input (scale [-1,1]) → thêm batch dim
    """
    # Bước 1: Convert về RGB (ảnh PNG có thể có kênh alpha RGBA)
    img = pil_img.convert("RGB")

    # Bước 2: Resize về đúng kích thước model yêu cầu
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Bước 3: Chuyển sang numpy array — shape: (224, 224, 3)
    img_array = np.array(img, dtype=np.float32)

    # Bước 4: preprocess_input của MobileNetV2
    # Scale từ [0,255] → [-1, 1] (đây là cách MobileNetV2 được train)
    img_array = preprocess_input(img_array)

    # Bước 5: Thêm batch dimension — shape: (1, 224, 224, 3)
    # Model luôn nhận input dạng batch, dù chỉ có 1 ảnh
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Giao diện

input_type = st.selectbox(
    "Chọn cách nhập ảnh:", ("Tải hình ảnh", "Sử dụng camera"), index=0
)
img = None # khời tạo biến ảnh

if input_type == "Sử dụng camera":
    camera_input = st.camera_input("Chụp ảnh chó hoặc mèo")
    if camera_input is not None:
        img = Image.open(camera_input)
        st.image(img, caption="Ảnh từ camera", use_container_width=True)
    else:
        st.info("📷 Nhấn nút camera để chụp ảnh")

else:  # Upload image
    uploaded_file = st.file_uploader(
        "Upload ảnh chó hoặc mèo (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh đã upload", use_container_width=True)
    else:
        st.info("📤 Chọn một file ảnh để bắt đầu")

# Predict

if img is not None:
    if st.button("🔍 Nhận diện", type="primary"):
        with st.spinner("Đang phân tích..."):
            # Tiền xử lý ảnh
            img_input = preprocess_image(img)

            # Dự đoán — model.predict trả về array shape (1, 2)
            # mỗi giá trị là xác suất thuộc về class đó
            predictions = model.predict(img_input, verbose=0)

            # argmax: lấy index có xác suất cao nhất
            prediction_idx = np.argmax(predictions[0])
            predicted_label = class_names[prediction_idx]
            confidence = float(predictions[0][prediction_idx])

        # Hiển thị kết quả chính
        if confidence >= 0.7:
            st.success(
                f"✅ **Kết quả:** `{predicted_label}` — Độ tin cậy: **{confidence*100:.1f}%**"
            )
        elif confidence >= 0.4:
            st.warning(
                f"⚠️ **Kết quả:** `{predicted_label}` — Độ tin cậy: **{confidence*100:.1f}%** (không chắc lắm)"
            )
        else:
            st.error(
                f"❌ **Kết quả:** `{predicted_label}` — Độ tin cậy: **{confidence*100:.1f}%** (rất không chắc)"
            )

        # Top 3 dự đoán
        st.subheader("📊 Top 3 dự đoán:")
        top3_idx = np.argsort(predictions[0])[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            label = class_names[idx]
            prob = float(predictions[0][idx])
            bar_color = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
            st.write(f"{bar_color} `{label}` — {prob*100:.2f}%")
            st.progress(prob)
else:
    if input_type == "Tải hình ảnh":
        st.warning("⚠️ Vui lòng upload ảnh trước khi nhận diện")