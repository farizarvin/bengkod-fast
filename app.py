import streamlit as st
import pandas as pd
import joblib 

# 1. Pengaturan Halaman
st.set_page_config(page_title="Aplikasi Prediksi Depresi", layout="wide", page_icon="🧠")

st.title("🧠 Aplikasi Prediksi Depresi Berbasis Machine Learning")
st.markdown("""
Aplikasi ini bertujuan untuk mendeteksi tingkat kerentanan seseorang terhadap depresi 
berdasarkan berbagai metrik akademik, finansial, rutinitas, dan latar belakang kesehatan.
""")

# Elemen Pendukung: Penjelasan Fitur
with st.expander("📚 Panduan & Penjelasan Lengkap Fitur"):
    st.markdown("""
    - **Academic Pressure**: Seberapa besar tekanan akademik yang sedang dihadapi (0 = Sangat Rendah, 5 = Sangat Tinggi).
    - **Have you ever had suicidal thoughts?**: Apakah Anda pernah memiliki pemikiran untuk mengakhiri hidup? (Yes/No).
    - **Financial Stress**: Seberapa besar stres Anda terkait kondisi keuangan (1 = Sangat Rendah, 5 = Sangat Tinggi).
    - **Age**: Umur Anda saat ini.
    - **Work/Study Hours**: Total jumlah jam yang dihabiskan untuk bekerja atau belajar dalam satu hari.
    - **Dietary Habits**: Kualitas kebiasaan makan sehari-hari (Healthy = Sehat, Moderate = Sedang, Unhealthy = Tidak Sehat).
    - **Study Satisfaction**: Tingkat kepuasan terhadap pencapaian belajar (0 = Sangat Tidak Puas, 5 = Sangat Puas).
    - **Sleep Duration**: Rata-rata durasi tidur per hari.
    - **Family History of Mental Illness**: Apakah ada riwayat penyakit mental di dalam keluarga inti Anda? (Yes/No).
    - **CGPA**: Nilai IPK / prestasi rata-rata (Skala 0.0 - 10.0).
    """)

# 2. Fungsi untuk memuat model
@st.cache_resource
def load_model_data():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None

model_data = load_model_data()

if model_data is not None:
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    mappings = model_data.get('mappings', {})
    daftar_fitur = model_data.get('selected_features', [])
    
    # 3. Definisi tipe input dan konfigurasinya (dilengkapi dengan Help/Tooltip)
    fitur_input_types = {
        'Academic Pressure': {'type': 'number', 'min': 0, 'max': 5, 'default': 3, 'help': 'Skala tekanan akademik (0-5)'},
        'Have you ever had suicidal thoughts ?': {'type': 'category', 'options': ['Yes', 'No'], 'help': 'Pernah terpikir mengakhiri hidup?'},
        'Financial Stress': {'type': 'number', 'min': 1.0, 'max': 5.0, 'default': 3.0, 'help': 'Skala tekanan finansial (1-5)'},
        'Age': {'type': 'number', 'min': 10, 'max': 100, 'default': 25, 'help': 'Umur (tahun)'},
        'Work/Study Hours': {'type': 'number', 'min': 0, 'max': 24, 'default': 5, 'help': 'Total jam kerja/belajar per hari'},
        'Dietary Habits': {'type': 'category', 'options': ['Healthy', 'Moderate', 'Unhealthy'], 'help': 'Pola makan'},
        'Study Satisfaction': {'type': 'number', 'min': 0, 'max': 5, 'default': 3, 'help': 'Kepuasan hasil belajar (0-5)'},
        'Sleep Duration': {'type': 'category', 'options': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'], 'help': 'Durasi tidur harian'},
        'Family History of Mental Illness': {'type': 'category', 'options': ['Yes', 'No'], 'help': 'Riwayat penyakit mental keluarga'},
        'CGPA': {'type': 'number', 'min': 0.0, 'max': 10.0, 'default': 7.0, 'help': 'Nilai IPK skala 10'}
    }

    st.write("---")
    
    # Layout aplikasi menggunakan kolom (Kiri: Form Input, Kanan: Hasil & Visualisasi)
    col1, col2 = st.columns([2, 1])

    with col1:
        dengan_form = st.form("form_prediksi")
        
        with dengan_form:
            st.subheader("📝 Input Data Diri & Akademik")
            input_pengguna = {}
            
            # Form dibagi menjadi 2 kolom agar lebih rapi
            c1, c2 = st.columns(2)
            
            for i, fitur in enumerate(daftar_fitur):
                config = fitur_input_types.get(fitur)
                # Bergantian kolom kiri (c1) dan kanan (c2)
                col_target = c1 if i % 2 == 0 else c2
                
                with col_target:
                    if config:
                        if config['type'] == 'number':
                            input_pengguna[fitur] = st.number_input(
                                label=f"{fitur}", 
                                min_value=config['min'], 
                                max_value=config['max'], 
                                value=config['default'],
                                help=config['help']
                            )
                        elif config['type'] == 'category':
                            input_pengguna[fitur] = st.selectbox(
                                label=f"{fitur}", 
                                options=config['options'],
                                help=config['help']
                            )
                    else:
                        input_pengguna[fitur] = st.text_input(label=f"{fitur}")
                        
            st.write("") # Spacing
            tombol_submit = st.form_submit_button("🔍 Analisis & Prediksi", use_container_width=True)

    with col2:
        st.subheader("📊 Hasil Prediksi")
        
        if not tombol_submit:
            st.info("Silakan isi data pada form di sebelah kiri dan tekan tombol 'Analisis & Prediksi' untuk melihat hasilnya.")
            
        if tombol_submit:
            if model is not None:
                # Konversi input format dictionary menjadi sebuah baris DataFrame
                fitur_df = pd.DataFrame([input_pengguna])
                
                # Menerapkan mapping yang tersimpan di model.pkl
                if 'sleep_mapping' in mappings and 'Sleep Duration' in fitur_df.columns:
                    fitur_df['Sleep Duration'] = fitur_df['Sleep Duration'].map(mappings['sleep_mapping'])
                if 'dietary_mapping' in mappings and 'Dietary Habits' in fitur_df.columns:
                    fitur_df['Dietary Habits'] = fitur_df['Dietary Habits'].map(mappings['dietary_mapping'])
                if 'binary_mapping' in mappings:
                    binary_maps = mappings['binary_mapping']
                    for col in binary_maps.keys():
                        if col in fitur_df.columns:
                            fitur_df[col] = fitur_df[col].map(binary_maps[col])
                            
                fitur_df = fitur_df.fillna(0)
                
                # Menerapkan Scaler jika tersedia
                if scaler is not None:
                    scaler_cols = list(scaler.feature_names_in_)
                    scaler_df = pd.DataFrame(columns=scaler_cols)
                    
                    for col in scaler_cols:
                        if col in fitur_df.columns:
                            scaler_df.loc[0, col] = fitur_df.loc[0, col]
                        else:
                            scaler_df.loc[0, col] = 0.0
                            
                    scaler_df_scaled = scaler.transform(scaler_df)
                    scaler_df_scaled = pd.DataFrame(scaler_df_scaled, columns=scaler_cols)
                    
                    for col in scaler_cols:
                        if col in fitur_df.columns:
                            fitur_df[col] = scaler_df_scaled[col]
                
                try:
                    # Menjalankan prediksi kelas dan probabilitas
                    hasil = model.predict(fitur_df)
                    
                    # Beberapa model klasifikasi mendukung fitur 'predict_proba' untuk persentase
                    if hasattr(model, "predict_proba"):
                        probabilitas = model.predict_proba(fitur_df)[0]
                        prob_depresi = probabilitas[1] * 100
                    else:
                        prob_depresi = 100.0 if (str(hasil[0]) == "1" or str(hasil[0]) == "1.0") else 0.0
                    
                    # 4. Visualisasi & Alert Hasil
                    if str(hasil[0]) == "1" or str(hasil[0]) == "1.0":
                        st.error("⚠️ **Peringatan:** Model mendeteksi kerentanan **DEPRESI**.")
                    else:
                        st.success("✅ **Aman:** Model mendeteksi **TIDAK DEPRESI**.")
                        
                    st.metric(label="Persentase Risiko Depresi", value=f"{prob_depresi:.1f}%")
                    st.progress(int(prob_depresi) // 1)
                    
                    st.write("---")
                    st.write("**Visualisasi Tingkat Stres & Kepuasan:**")
                    
                    # Membuat DataFrame khusus untuk Bar Chart visualisasi tekanan
                    vis_data = pd.DataFrame({
                        'Faktor': ['Academic Pressure', 'Financial Stress', 'Study Satisfaction'],
                        'Skor': [
                            input_pengguna.get('Academic Pressure', 0), 
                            input_pengguna.get('Financial Stress', 0), 
                            input_pengguna.get('Study Satisfaction', 0)
                        ]
                    }).set_index('Faktor')
                    
                    st.bar_chart(vis_data, height=200)
                    st.caption("Skor tinggi pada tekanan akademik dan finansial dapat berpotensi memicu depresi. Sebaliknya, kepuasan belajar yang baik adalah indikator positif.")
                    
                except Exception as error_msg:
                    st.error(f"Terjadi error saat memprediksi data: {error_msg}")
            else:
                st.error("Gagal menjalankan prediksi: Model belum termuat.")
else:
    st.error("Gagal memuat model.pkl atau format tidak sesuai (bukan dictionary).")
