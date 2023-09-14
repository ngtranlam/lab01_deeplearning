import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn import datasets

# Tải mô hình
@st.cache_data
def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

# Danh sách mô hình
models = {
    "Decision Tree": load_model("Decision_Tree.pkl"),
    "Naive Bayes": load_model("Naive_Bayes.pkl"),
    "SVM": load_model("SVM.pkl"),
    "XGBoost": load_model("XGBoost.pkl")
}


st.markdown("""
    <style>
        .reportview-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .main .block-container {
            width: 65%;
        }
        h1 {
            color: #ff6347;
            font-family: 'Courier New', Courier, monospace;
        }
        .prediction {
            padding: 20px;
                border: 2px solid #ff6347;
                border-radius: 5px;
            font-size: 20px;
            text-align: center;
            background-color: #e6e6e6;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Phân loại hạt giống")

model_choice = st.sidebar.selectbox("Chọn mô hình:", list(models.keys()))
st.sidebar.subheader("Nhập dữ liệu:")
data_input_method = st.sidebar.radio("Lựa chọn phương thức nhập dữ liệu:", ["Nhập thủ công", "Tải file CSV"])

if data_input_method == "Tải file CSV":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV:", type=['csv'])
    if uploaded_file:
        data = load_csv(uploaded_file)
        st.sidebar.write(data)
        user_data = data.values
    else:
        user_data = None
else:
    use_slider = st.sidebar.checkbox("Sử dụng thanh kéo")

    if use_slider:
        acreage = st.sidebar.slider("Acreage", 10.0, 25.0, 15.0)
        perimeter = st.sidebar.slider("Perimeter", 10.0, 25.0, 15.0)
        compactness = st.sidebar.slider("Compactness", 0.7, 1.0, 0.85)
        length_of_kernel = st.sidebar.slider("Length of Kernel", 4.0, 7.0, 5.5)
        width_of_kernel = st.sidebar.slider("Width of Kernel", 2.0, 5.0, 3.5)
        asymmetry_coefficient = st.sidebar.slider("Asymmetry Coefficient", 1.0, 10.0, 5.5)
        length_of_kernel_groove = st.sidebar.slider("Length of Kernel Groove", 4.0, 7.0, 5.5)
    else:
        # Các trường nhập dữ liệu nếu không dùng slider
        acreage = st.sidebar.number_input("Acreage", 10.0, 25.0, 15.0)
        perimeter = st.sidebar.number_input("Perimeter", 10.0, 25.0, 15.0)
        compactness = st.sidebar.number_input("Compactness", 0.7, 1.0, 0.85)
        length_of_kernel = st.sidebar.number_input("Length of Kernel", 4.0, 7.0, 5.5)
        width_of_kernel = st.sidebar.number_input("Width of Kernel", 2.0, 5.0, 3.5)
        asymmetry_coefficient = st.sidebar.number_input("Asymmetry Coefficient", 1.0, 10.0, 5.5)
        length_of_kernel_groove = st.sidebar.number_input("Length of Kernel Groove", 4.0, 7.0, 5.5)
    
    user_data = np.array([
        acreage, 
        perimeter, 
        compactness, 
        length_of_kernel, 
        width_of_kernel, 
        asymmetry_coefficient, 
        length_of_kernel_groove
    ]).reshape(1, -1)
    
if user_data is not None:
    model = models[model_choice]
    prediction = model.predict(user_data)
    
    st.subheader(f"Kết quả dự đoán từ thuật toán {model_choice}:")
    st.markdown(f"<div class='prediction'>{prediction[0]}</div>", unsafe_allow_html=True)
