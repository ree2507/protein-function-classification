# Perbandingan Arsitektur CNN, LSTM, dan Model Pretrained ESM-2 untuk Klasifikasi Fungsi Protein

## Deskripsi Project

Project ini bertujuan untuk membandingkan tiga pendekatan deep learning dalam klasifikasi fungsi protein berdasarkan sekuens asam amino. Klasifikasi fungsi protein merupakan tugas fundamental dalam bioinformatika yang dapat mempercepat penemuan obat dan penelitian biologi. Tiga arsitektur yang dibandingkan mewakili tiga generasi pendekatan AI: CNN (computer vision approach), LSTM (sequential/NLP approach), dan ESM-2 (large language model / transfer learning approach).

## Masalah

- **Keterbatasan alignment-based methods:** Metode klasifikasi protein tradisional seperti BLAST bergantung pada sequence alignment yang lambat dan tidak scalable untuk dataset besar.
- **Kebutuhan komputasi tinggi:** Model deep learning modern seperti ESM-2 membutuhkan resource komputasi yang besar, belum tentu cocok untuk hardware terbatas.
- **Trade-off arsitektur:** Belum diketahui secara kuantitatif bagaimana perbandingan performa antara CNN (ringan), LSTM (sedang), dan ESM-2 (berat) dalam konteks klasifikasi fungsi protein pada dataset terbatas.
- **Ketidakseimbangan performa per kelas:** Beberapa kelas protein (terutama Hydrolase) secara konsisten memiliki F1-score lebih rendah dari kelas lain.

## Tujuan

- Membandingkan local feature extraction (CNN) vs sequential modeling (LSTM) vs evolutionary scale modeling (ESM-2) untuk klasifikasi fungsi protein.
- Mengoptimalkan pelatihan model untuk hardware lokal.
- Mengidentifikasi trade-off antara kompleksitas model, kebutuhan komputasi, dan performa prediktif.
- Meningkatkan performa klasifikasi pada kelas-kelas yang sulit (seperti Hydrolase).

## Dataset

### Sumber
Dataset diperoleh dari **UniProt** (Universal Protein Resource) melalui REST API, khususnya dari basis data **Swiss-Prot** (reviewed/annotated manually). Swiss-Prot dipilih karena kualitas anotasinya yang tinggi dan terverifikasi.

### Detail Dataset
- **Total sekuens:** ~25.000 sekuens (setelah stratified sampling).
- **Split:** 80% Train (~20.000), 20% Test (~5.000).
- **Jumlah kelas:** 6 keluarga protein.
- **Kelas protein:**
  | Kelas | Deskripsi | Support (Test) |
  |-------|-----------|:--------------:|
  | GPCR | G protein-coupled receptor | 664 |
  | Hydrolase | Enzim hidrolase | 857 |
  | Ion Channel | Saluran ion | 894 |
  | Kinase | Enzim kinase | 837 |
  | Oxidoreductase | Enzim oksidoreduktase | 906 |
  | Transcription Factor | Faktor transkripsi | 861 |
- **Rentang panjang sekuens:** Bervariasi dari puluhan hingga ribuan asam amino (max length dipotong ke 1000 untuk efisiensi komputasi).
- **Kriteria seleksi:** Reviewed (Swiss-Prot), organisme *Homo sapiens*, panjang sekuens minimal 50 asam amino.

## Arsitektur dan Model

### 1. CNN (Convolutional Neural Network)
- **Pendekatan:** Multi-scale 1D-CNN dengan 3 branch konvolusi (kernel 3, 5, dan 9) untuk menangkap motif protein dengan panjang berbeda.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Spatial Dropout (p=0.3) setelah embedding, Batch Normalization, Weight Decay.
- **Parameter:** ~1,5 juta.
- **Karakteristik:** Cepat dilatih, fokus pada pola lokal/motif.

### 2. Bidirectional LSTM (Long Short-Term Memory)
- **Pendekatan:** Dua layer LSTM bidirectional dengan hidden size 128, menangkap long-range dependencies dalam sekuens.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Embedding Dropout (p=0.2), Recurrent Dropout (p=0.4), FC Dropout (p=0.6), Weight Decay (1e-4), Gradient Clipping.
- **Parameter:** ~2,1 juta.
- **Karakteristik:** Menangani ketergantungan jarak jauh, memahami "grammar" urutan protein.

### 3. ESM-2 (Evolutionary Scale Modeling)
- **Pendekatan:** Fine-tuning model pretrained `esm2_t12_35M_UR50D` (35M parameter) dari Meta AI.
- **Strategi:** Feature extraction + fine-tuning pada klasifikasi 6 kelas.
- **Karakteristik:** Transfer learning dari 65 juta sekuens protein, potensi akurasi tertinggi dengan komputasi tertinggi.

## Tahapan Pengerjaan

### Phase 1: Environment Setup
- Inisialisasi Git dan konfigurasi `.gitignore`.
- Setup virtual environment (`.venv`) dengan CUDA-enabled PyTorch.
- Registrasi kernel Jupyter untuk deteksi CUDA.

### Phase 2: Data Acquisition (`01_data_acquisition.ipynb`)
- Fetching sekuens protein dan label keluarga dari UniProt REST API.
- Menyaring data reviewed (Swiss-Prot) dengan stratified sampling.
- Menyimpan data mentah ke `data/raw/`.

### Phase 3: Preprocessing (`02_preprocessing.ipynb`)
- Pembersihan sekuens (handle karakter non-standar).
- Encoding label dan tokenisasi asam amino ke integer.
- Analisis distribusi panjang sekuens.
- Stratified split (Train/Validation/Test).
- Menyimpan data processed ke `data/processed/`.

### Phase 4: Model Implementation
- **Phase 4a:** CNN (`03_cnn_model.ipynb`) — Multi-scale 1D-CNN.
- **Phase 4b:** LSTM (`04_lstm_model.ipynb`) — Bidirectional LSTM.
- **Phase 4c:** ESM-2 (`05_esm2_model.ipynb`) — Fine-tuning ESM-2.
- Setiap model dilatih dengan 50 epoch, early stopping (patience=5), dan menyimpan checkpoint terbaik.
- Evaluasi: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix, Training History plots.

### Phase 5: Comparative Analysis (`06_comparison_analysis.ipynb`)
- Perbandingan metrik performa antar model (Accuracy, F1, MCC).
- Visualisasi perbandingan (bar charts, confusion matrices).
- Analisis trade-off performa vs komputasi.
- Kesimpulan dan rekomendasi arsitektur terbaik.

## Ekspektasi Output

- **Notebook Jupyter (.ipynb):** Untuk setiap fase pengerjaan (data acquisition, preprocessing, 3 model, dan analisis komparatif).
- **Dataset:** Data mentah (`data/raw/`) dan data siap-pakai (`data/processed/`).
- **Model Checkpoints:** Bobot model terbaik untuk CNN, LSTM, dan ESM-2 (`models/`).
- **Metrik Evaluasi:** File JSON berisi accuracy, F1-score, MCC, dan history pelatihan (`data/results/`).
- **Visualisasi:**
  - Kurva loss dan accuracy per model (train vs validation).
  - Confusion matrix untuk setiap model.
  - Bar chart perbandingan metrik antar model.
- **Laporan Komparatif:**
  - Tabel perbandingan performa (accuracy, F1 macro, MCC, training time).
  - Matriks korelasi antar model.
  - Analisis kelebihan dan kekurangan masing-masing arsitektur dalam konteks klasifikasi protein.

## Manfaat

- **Referensi pemilihan arsitektur:** Memberikan panduan kuantitatif bagi peneliti bioinformatika dalam memilih arsitektur deep learning yang tepat berdasarkan trade-off akurasi vs sumber daya komputasi.
- **Validasi ESM-2 untuk dataset kecil:** Menguji apakah model pretrained berskala besar (ESM-2) memberikan peningkatan signifikan dibanding arsitektur yang lebih sederhana pada dataset protein dengan jumlah terbatas.
- **Optimasi hardware menengah:** Menunjukkan strategi optimasi (gradient clipping, mixed precision, batch size tuning) yang efektif untuk GPU dengan VRAM terbatas (4GB).
- **Dasar pengembangan sistem klasifikasi protein:** Hasil perbandingan dapat digunakan sebagai dasar pengembangan pipeline klasifikasi fungsi protein otomatis untuk aplikasi penemuan obat dan annotasi genom.
- **Reproducible research:** Seluruh kode, konfigurasi, dan dataset terdokumentasi dengan baik untuk memudahkan reproduksi dan pengembangan lebih lanjut oleh peneliti lain.
