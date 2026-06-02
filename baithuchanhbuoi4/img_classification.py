import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# -- CẤU HÌNH TRANG --
st.set_page_config(page_title="Image Classification with History", layout="centered")
st.title("Image Classification with History")

# -- TẢI MÔ HÌNH --
@st.cache_resource
def load_my_model():
    model_path = Path(__file__).resolve().parents[1] / "checkpoint 1" / "d&c_classification_model_best.keras"
    return load_model(model_path)

model = load_my_model()

NUM_CLASSES = 10  # Số lớp, ví dụ MNIST có 10 chữ số

# -- KHỞI TẠO SESSION STATE --
# Lưu lịch sử dự đoán
if "history" not in st.session_state:
    st.session_state.history = []  # mỗi phần tử là dict: {"image_name": str, "pred_class": int, "image": PIL}

# Lưu trang hiện tại để phân trang
if "page_num" not in st.session_state:
    st.session_state.page_num = 0

# Số item trên mỗi trang
ITEMS_PER_PAGE = 3

# -- HÀM TIỀN XỬ LÝ ẢNH --
def preprocess_image(uploaded_file):
    """Chuyển ảnh upload thành dạng numpy array (1, 224, 224, 3) cho model RGB input."""
    image = Image.open(uploaded_file).convert('RGB')  # chuyển sang RGB
    image = image.resize((224, 224))   # resize về 224x224 theo yêu cầu model
    img_array = keras_image.img_to_array(image)    # (224,224,3)
    img_array = img_array / 255.0                  # chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    return img_array, image

# -- DỰ ĐOÁN --
def predict_image(uploaded_file):
    img_array, pil_img = preprocess_image(uploaded_file)
    preds = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(preds[0])
    return predicted_class, pil_img

# -- GIAO DIỆN --
# 1. Upload ảnh
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# 2. Nút Classify
if st.button("Classify"):
    if uploaded_file is not None:
        with st.spinner("Đang dự đoán..."):
            pred_class, img = predict_image(uploaded_file)
        
        # Hiển thị kết quả
        st.success(f"**Predicted Class: {pred_class}**")
        
        # Lưu vào lịch sử
        st.session_state.history.insert(0, {
            "image_name": uploaded_file.name,
            "pred_class": pred_class,
            "image": img
        })
        
        # Reset về trang đầu tiên để thấy kết quả mới nhất
        st.session_state.page_num = 0
    else:
        st.warning("Vui lòng tải lên một ảnh trước.")

# 3. Hiển thị lịch sử dự đoán
st.subheader("Classification History")

if len(st.session_state.history) == 0:
    st.info("Chưa có lịch sử dự đoán nào. Hãy upload ảnh và nhấn Classify.")
else:
    # Tính tổng số trang
    total_items = len(st.session_state.history)
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    
    # Lấy các item của trang hiện tại
    start_idx = st.session_state.page_num * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    current_items = st.session_state.history[start_idx:end_idx]
    
    # Tạo dataframe hiển thị
    data = []
    for item in current_items:
        data.append({
            "Image Name": item["image_name"],
            "Predicted Class": item["pred_class"]
        })
    df = pd.DataFrame(data)
    st.table(df)   # hoặc st.dataframe(df)
    
    # Phân trang
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Trang trước") and st.session_state.page_num > 0:
            st.session_state.page_num -= 1
            st.rerun()
    with col2:
        if st.button("Trang sau ▶") and st.session_state.page_num < total_pages - 1:
            st.session_state.page_num += 1
            st.rerun()
    
    st.caption(f"Trang {st.session_state.page_num + 1} / {total_pages}")

# 4. Tùy chọn xóa lịch sử
if st.button("🗑️ Xóa toàn bộ lịch sử"):
    st.session_state.history.clear()
    st.session_state.page_num = 0
    st.rerun()