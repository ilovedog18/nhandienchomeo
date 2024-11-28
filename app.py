import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tải mô hình đã huấn luyện
from tensorflow.keras.models import load_model
model = load_model("C:/Users/hoang/Desktop/nhan dien cho meo/KHDL.keras")

# Hàm chuẩn bị ảnh đầu vào cho mô hình
def prepare_image(img):
    img = img.resize((256, 256))  # Đảm bảo kích thước ảnh giống như trong quá trình huấn luyện
    img_array = np.array(img) # Chuẩn hóa ảnh (nếu bạn đã chuẩn hóa trong quá trình huấn luyện)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    return img_array

# Giao diện người dùng
st.title('Nhận diện Chó và Mèo')

# Tải ảnh từ máy tính
uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đầu vào", use_column_width=True)
    
    # Chuẩn bị ảnh và dự đoán
    img_array = prepare_image(img)
    prediction = model.predict(img_array)
    
    # Xử lý kết quả dự đoán
    # class_names = ["Cat", "Dog"]  # Giả sử lớp 0 là mèo, lớp 1 là chó
    # predicted_class = class_names[np.argmax(prediction)]  # Lấy lớp dự đoán có xác suất cao nhất
    # confidence = np.max(prediction)  # Xác suất lớn nhất
    confidence = prediction[0][0]
    if confidence < 0:
        predicted_class = "Đây là Mèo"
    else:
        predicted_class = "Đây là Chó"
    # Hiển thị kết quả
    st.write(f"Kết quả dự đoán: *{predicted_class}*")
