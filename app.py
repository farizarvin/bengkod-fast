import streamlit as st
import pandas as pd
import joblib 

# 1. Pengaturan Halaman
st.set_page_config(page_title="Aplikasi Prediksi Machine Learning", layout="centered")
st.title("Aplikasi Prediksi Berbasis Web")
st.write("Silahkan masukan variabel pada form di bawah ini, lalu tekan tombol prediksi.")

# 2. Fungsi untuk memuat data (di-cache agar lebih cepat)
@st.cache_data
def load_data():
    # SESUAIKAN DENGAN NAMA FILE CSV ANDA
    try:
        return pd.read_csv("dataset.csv") 
    except FileNotFoundError:
        return None

# 3. Fungsi untuk memuat model (di-cache agar tidak berulang kali memuat model)
@st.cache_resource
def load_model():
    # SESUAIKAN DENGAN NAMA FILE MODEL ANDA (pickle / joblib)
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None

df = load_data()
model = load_model()

if df is not None:
    # 4. Filter kolom: Tentukan nama kolom target (hasil/label) dari dataset
    # agar tidak dimasukkan sebagai bagian dari input form
    nama_kolom_target = 'target' # GANTI INI DENGAN NAMA KOLOM LABEL/TARGET ANDA
    
    if nama_kolom_target in df.columns:
        daftar_fitur = df.drop(columns=[nama_kolom_target]).columns
    else:
        daftar_fitur = df.columns

    # 5. Membuat Form Interaktif
    dengan_form = st.form("form_prediksi")
    
    with dengan_form:
        st.subheader("Input Data Prediksi")
        
        input_pengguna = {}
        
        # Looping dinamis untuk membuat kolom input. 
        # Jika fiturnya angka -> st.number_input. 
        # Jika kata/kategori -> st.selectbox.
        for fitur in daftar_fitur:
            if pd.api.types.is_numeric_dtype(df[fitur]):
                # Kita gunakan median sebagai nilai default
                default_val = float(df[fitur].median())
                input_pengguna[fitur] = st.number_input(label=f"Masukkan {fitur}", value=default_val)
            else:
                # Kumpulkan tipe nilai yang unik dalam list sebagai opsi selectbox
                opsi = df[fitur].dropna().unique().tolist()
                input_pengguna[fitur] = st.selectbox(label=f"Pilih {fitur}", options=opsi)
                
        # Tombol action di dalam form
        tombol_submit = st.form_submit_button("Jalankan Prediksi")
        
        # 6. Pemrosesan Hasil dari Input Prediksi
        if tombol_submit:
            if model is not None:
                # Konversi input format dictionary menjadi sebuah baris DataFrame
                fitur_df = pd.DataFrame([input_pengguna])
                
                try:
                    # Jalankan prediksi
                    hasil = model.predict(fitur_df)
                    
                    # Coba tampilkan hasil
                    st.success("Proses Prediksi Berhasil!")
                    st.write("### Hasil Akhir:")
                    st.title(f"{hasil[0]}")
                    
                except Exception as error_msg:
                    st.error(f"Terjadi error saat load model / memprediksi: {error_msg}")
            else:
                st.error("Gagal menjalankan prediksi: Model (misal: model.pkl) tidak ditemukan.")
else:
    st.warning("⚠️ File dataset.csv tidak ditemukan. Mohon pastikan file dataset Anda berada di folder yang sama dengan script.")
