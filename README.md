Machine Learning Project: Random Forest, PSO, and SMOTE
Proyek ini mengimplementasikan pendekatan Machine Learning yang menggunakan metode Random Forest untuk klasifikasi, Particle Swarm Optimization (PSO) untuk seleksi fitur, dan SMOTE untuk menangani ketidakseimbangan data. Proyek ini dilengkapi dengan antarmuka pengguna yang interaktif menggunakan Streamlit.

📋 Deskripsi Proyek
Proyek ini bertujuan untuk memberikan solusi end-to-end dalam menangani data yang tidak seimbang, memilih fitur yang paling signifikan, dan mengklasifikasikan data dengan performa optimal. Dengan bantuan antarmuka Streamlit, pengguna dapat menjalankan analisis ini dengan mudah.

🛠️ Fitur Utama
Random Forest Classifier: Model berbasis ensemble yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi klasifikasi.
PSO (Particle Swarm Optimization): Algoritma optimasi berbasis populasi yang digunakan untuk seleksi fitur.
SMOTE (Synthetic Minority Over-sampling Technique): Teknik oversampling untuk menangani ketidakseimbangan dalam data.
Streamlit: Antarmuka pengguna berbasis web yang memudahkan interaksi dengan model.

📂 Struktur Proyek
project-root/
│
├── app.py              # File utama untuk Streamlit
├── utils.py            # Fungsi utilitas untuk pemrosesan data dan model
├── requirements.txt    # Daftar dependensi Python
├── data/               # Folder untuk dataset
├── models/             # Folder untuk menyimpan model yang dilatih (opsional)
└── README.md           # File README ini

🛠️ Instalasi dan Setup
Clone Repositori
git clone https://github.com/username/repo-name.git
cd repo-name

Buat dan Aktifkan Virtual Environment (Opsional tetapi disarankan)
python -m venv env
source env/bin/activate  # Untuk macOS/Linux
env\Scripts\activate     # Untuk Windows

Instal Dependensi
pip install -r requirements.txt

Pastikan requirements.txt Anda mencantumkan dependensi berikut:
streamlit
scikit-learn
imbalanced-learn
pyswarms
pandas
numpy

🚀 Cara Menjalankan Proyek
Buka Terminal dan arahkan ke direktori proyek.
Jalankan Aplikasi Streamlit

streamlit run app.py
Antarmuka pengguna akan terbuka di browser Anda. Anda dapat mengunggah dataset dan menyesuaikan parameter untuk menjalankan analisis.


