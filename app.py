import streamlit as st
import pandas as pd
import joblib 

# 1. Pengaturan Halaman
st.set_page_config(page_title="Aplikasi Prediksi Machine Learning", layout="centered")
st.title("Aplikasi Prediksi Berbasis Web")
st.write("Silahkan masukan variabel pada form di bawah ini, lalu tekan tombol prediksi.")

# 2. Fungsi untuk memuat model (di-cache agar tidak berulang kali memuat model)
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None

model = load_model()

# 3. Daftar 10 fitur yang digunakan oleh model
daftar_fitur = [
    'Academic Pressure', 
    'Have you ever had suicidal thoughts ?', 
    'Financial Stress', 
    'Age', 
    'Work/Study Hours', 
    'Dietary Habits', 
    'Study Satisfaction', 
    'Sleep Duration', 
    'Family History of Mental Illness', 
    'CGPA'
]

# Definisi tipe input untuk masing-masing fitur
fitur_input_types = {
    'Academic Pressure': {'type': 'number', 'min': 0.0, 'max': 5.0, 'default': 3.0},
    'Have you ever had suicidal thoughts ?': {'type': 'category', 'options': ['Yes', 'No']},
    'Financial Stress': {'type': 'number', 'min': 1.0, 'max': 5.0, 'default': 3.0},
    'Age': {'type': 'number', 'min': 10.0, 'max': 100.0, 'default': 25.0},
    'Work/Study Hours': {'type': 'number', 'min': 0.0, 'max': 24.0, 'default': 5.0},
    'Dietary Habits': {'type': 'category', 'options': ['Healthy', 'Moderate', 'Unhealthy', 'Others']},
    'Study Satisfaction': {'type': 'number', 'min': 0.0, 'max': 5.0, 'default': 3.0},
    'Sleep Duration': {'type': 'category', 'options': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']},
    'Family History of Mental Illness': {'type': 'category', 'options': ['Yes', 'No']},
    'CGPA': {'type': 'number', 'min': 0.0, 'max': 10.0, 'default': 7.0}
}

# 4. Membuat Form Interaktif
dengan_form = st.form("form_prediksi")

with dengan_form:
    st.subheader("Input Data Prediksi")
    
    input_pengguna = {}
    
    # Looping dinamis untuk membuat kolom input sesuai definisi di atas
    for fitur in daftar_fitur:
        config = fitur_input_types.get(fitur)
        if config:
            if config['type'] == 'number':
                input_pengguna[fitur] = st.number_input(
                    label=f"Masukkan {fitur}", 
                    min_value=float(config['min']), 
                    max_value=float(config['max']), 
                    value=float(config['default'])
                )
            elif config['type'] == 'category':
                input_pengguna[fitur] = st.selectbox(
                    label=f"Pilih {fitur}", 
                    options=config['options']
                )
        else:
            # Fallback jika fitur tidak didefinisikan secara khusus
            input_pengguna[fitur] = st.text_input(label=f"Masukkan {fitur}")
            
    # Tombol action di dalam form
    tombol_submit = st.form_submit_button("Jalankan Prediksi")
    
    # 5. Pemrosesan Hasil dari Input Prediksi
    if tombol_submit:
        if model is not None:
            # Konversi input format dictionary menjadi sebuah baris DataFrame
            fitur_df = pd.DataFrame([input_pengguna])
            
            

            try:
                # Jalankan prediksi
                hasil = model.predict(fitur_df)
                
                # Tampilkan hasil
                st.success("Proses Prediksi Berhasil!")
                st.write("### Hasil Prediksi Kelas (Depression):")
                st.title(f"{hasil[0]}")
                
            except Exception as error_msg:
                st.error(f"Terjadi error saat load model / memprediksi: {error_msg}")
        else:
            st.error("Gagal menjalankan prediksi: Model (model.pkl) tidak ditemukan.")
