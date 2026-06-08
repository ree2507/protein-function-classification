# Perbandingan Arsitektur CNN, LSTM, dan Model Pretrained ESM-2 untuk Klasifikasi Fungsi Protein

## Status Pengerjaan

Proyek masih dalam tahap pengembangan. Fase 1–5 (data acquisition, preprocessing, CNN v2, LSTM v5, ESM-2 v1, comparative analysis) dan integrasi Telegram Bot telah selesai. Fase pengembangan model lanjutan masih direncanakan (lihat `planning/`).

## Deskripsi Project

Project ini bertujuan untuk membandingkan tiga pendekatan deep learning dalam klasifikasi fungsi protein berdasarkan sekuens asam amino. Klasifikasi fungsi protein merupakan tugas fundamental dalam bioinformatika yang dapat mempercepat penemuan obat dan penelitian biologi. Tiga arsitektur yang dibandingkan mewakili tiga generasi pendekatan AI: CNN (computer vision approach), LSTM (sequential/NLP approach), dan ESM-2 (large language model / transfer learning approach).

## Masalah

- **Keterbatasan alignment-based methods:** Metode klasifikasi protein tradisional seperti BLAST bergantung pada sequence alignment yang lambat dan tidak scalable untuk dataset besar.
- **Kebutuhan komputasi tinggi:** Model deep learning modern seperti ESM-2 membutuhkan resource komputasi yang besar, belum tentu cocok untuk hardware terbatas.
- **Trade-off arsitektur:** Belum diketahui secara kuantitatif bagaimana perbandingan performa antara CNN (ringan), LSTM (sedang), dan ESM-2 (berat) dalam konteks klasifikasi fungsi protein pada dataset terbatas.
- **Ketidakseimbangan performa per kelas:** Beberapa kelas protein (terutama Hydrolase) secara konsisten memiliki F1-score lebih rendah dari kelas lain.

## Urgensi

Pemilihan arsitektur deep learning yang tepat untuk klasifikasi protein sangat bergantung pada trade-off antara akurasi, kecepatan, dan sumber daya komputasi — namun belum ada studi komparatif yang mengkuantifikasi trade-off ini secara langsung pada dataset protein dengan hardware menengah. Penelitian ini mengisi celah tersebut dengan membandingkan tiga generasi arsitektur secara sistematis, memberikan panduan kuantitatif bagi peneliti bioinformatika dengan keterbatasan komputasi.

## Tujuan

- Membandingkan local feature extraction (CNN) vs sequential modeling (LSTM) vs evolutionary scale modeling (ESM-2) untuk klasifikasi fungsi protein.
- Mengoptimalkan pelatihan model untuk hardware lokal.
- Mengidentifikasi trade-off antara kompleksitas model, kebutuhan komputasi, dan performa prediktif.
- Meningkatkan performa klasifikasi pada kelas-kelas yang sulit (seperti Hydrolase).

## Manfaat

- **Referensi pemilihan arsitektur:** Memberikan panduan kuantitatif dalam memilih arsitektur deep learning berdasarkan trade-off akurasi vs sumber daya komputasi.
- **Validasi ESM-2 untuk dataset kecil:** Menguji apakah model pretrained berskala besar memberikan peningkatan signifikan pada dataset protein dengan jumlah terbatas.
- **Optimasi hardware menengah:** Menunjukkan strategi optimasi (gradient clipping, mixed precision, batch size tuning) yang efektif untuk GPU dengan VRAM terbatas (4GB).
- **Dasar pengembangan sistem klasifikasi protein:** Hasil perbandingan dapat digunakan sebagai dasar pengembangan pipeline klasifikasi fungsi protein otomatis.

## Dataset

### Sumber
Dataset diperoleh dari **UniProt** (Universal Protein Resource) melalui REST API, khususnya dari basis data **Swiss-Prot** (reviewed/annotated manually).

### Detail Dataset
- **Total sekuens:** ~25.000 sekuens (setelah stratified sampling).
- **Split:** 70% Train (17,564), 15% Validation (3,764), 15% Test (3,764).
- **Jumlah kelas:** 6 keluarga protein.
- **Kelas protein:**

  | Kelas | Deskripsi | Support (Test) |
  |-------|-----------|:--------------:|
  | GPCR | G protein-coupled receptor | 498 |
  | Hydrolase | Enzim hidrolase | 643 |
  | Ion Channel | Saluran ion | 671 |
  | Kinase | Enzim kinase | 628 |
  | Oxidoreductase | Enzim oksidoreduktase | 679 |
  | Transcription Factor | Faktor transkripsi | 645 |

- **Rentang panjang sekuens:** Bervariasi dari puluhan hingga ribuan asam amino (max length dipotong ke 1000 untuk efisiensi komputasi).
- **Kriteria seleksi:** Reviewed (Swiss-Prot), organisme *Homo sapiens*, panjang sekuens minimal 50 asam amino.

## Arsitektur dan Model

### 1. CNN v2 (Convolutional Neural Network)
- **Pendekatan:** Multi-scale 1D-CNN dengan 3 branch konvolusi (kernel 3, 5, dan 9) — menangkap motif protein dengan panjang berbeda.
- **Pooling:** Combined Global Max + Average Pooling (6 × 128 = 768 fitur).
- **Regularisasi:** Spatial Dropout (p=0.3) setelah embedding, Batch Normalization, Weight Decay (1e-4).
- **Parameter:** ~340rb.
- **Training:** Lokal (RTX 2050 4GB), ~24s/epoch.
- **Hasil:** Accuracy 82.76%, F1 Macro 0.8349, MCC 0.7932.
- Detail optimasi: [`planning/cnn_planning.md`](planning/cnn_planning.md).

### 2. LSTM v5 (Bidirectional Long Short-Term Memory)
- **Pendekatan:** Dua layer LSTM bidirectional dengan hidden size 128 — menangkap long-range dependencies dalam sekuens.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Embedding Dropout (p=0.2), Recurrent Dropout (p=0.4), FC Dropout (p=0.5 + 0.3), Weight Decay (5e-5), Gradient Clipping.
- **Parameter:** ~827rb.
- **Training:** Lokal (RTX 2050 4GB), ~283s/epoch.
- **Optimasi:** 5 iterasi (v1→v5) — Focal Loss γ=1.0, Label Smoothing ε=0.1, Embedding 128, Xavier/Orthogonal/Kaiming weight init.
- **Hasil:** Accuracy 86.85%, F1 Macro 0.8716, MCC 0.8422.
- Detail optimasi: [`planning/lstm_planning.md`](planning/lstm_planning.md).

### 3. ESM-2 v1 (Evolutionary Scale Modeling)
- **Pendekatan:** LoRA fine-tuning model pretrained `esm2_t12_35M_UR50D` (35M parameter) dari Meta AI pada query/key/value projections (r=8, alpha=32, dropout=0.1).
- **Classifier Head:** MLP 480→256→6.
- **Parameter:** ~34,3M total (124,7rb trainable).
- **Training:** Google Colab T4 (15.6 GB VRAM) — RTX 2050 4GB terbukti terlalu lambat (~30 menit/epoch).
- **Hyperparameter:** batch=16, accum=2, effective batch=32, FP16, AdamW (lr=1e-4).
- **Hasil:** Accuracy 93.09%, F1 Macro 0.9327, MCC 0.9171.
- Detail optimasi: [`planning/esm2_planning.md`](planning/esm2_planning.md).

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
- Stratified split (Train/Validation/Test) — rasio 70:15:15.
- Menyimpan data processed ke `data/processed/`.

### Phase 4: Model Implementation
- **Phase 4a:** CNN (`03_cnn_model.ipynb`) — Multi-scale 1D-CNN v2.
- **Phase 4b:** LSTM (`04_lstm_model.ipynb`) — Bidirectional LSTM v5.
- **Phase 4c:** ESM-2 (`05_esm2_model_colab.ipynb`) — LoRA fine-tuning ESM-2 v1 di Google Colab T4.
- Setiap model: 50 epoch, early stopping (patience=5), checkpoint terbaik.
- Evaluasi: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix, Training History.

### Phase 5: Comparative Analysis (`06_comparative_analysis.ipynb`)
- Load model weights CNN + LSTM + ESM-2.
- Perbandingan metrik performa dan visualisasi (bar charts, confusion matrices, radar chart).
- Analisis error, model agreement, dan trade-off performa vs komputasi.
- Menyimpan prediksi ke `data/results/comparative_predictions.npz`.

### Phase 6: Telegram Bot Integration
- **Backend API:** FastAPI (`api/main.py`) — endpoint `/predict`, `/health`, `/models`.
- **Bot Telegram:** `telegram_bot.py` — 6 command handler (`/start`, `/help`, `/about`, `/compare`, `/model`) + auto-detect sequence.
- **Model Serving:** Lazy loading dengan caching via `api/model_loader.py`.
- **Output Edukatif:** Setiap prediksi menampilkan deskripsi famili, fungsi molekuler, proses biologis, lokasi sel, dan rincian probabilitas.
<!-- - **Cara menjalankan:** `python run.py` (menjalankan API + Bot bersamaan). -->
- Detail implementasi: [`planning/rancangan_telegram.md`](planning/rancangan_telegram.md).

## Output

- **Notebook Jupyter (.ipynb):** Setiap fase pengerjaan (01–06).
- **Dataset:** Data mentah (`data/raw/`) dan data siap-pakai (`data/processed/`).
- **Model Checkpoints:** Bobot model terbaik (`models/`).
- **Metrik Evaluasi:** File JSON accuracy, F1-score, MCC, history pelatihan (`data/results/`).
- **Prediksi:** File NPZ prediksi + probabilitas 3 model (`data/results/comparative_predictions.npz`).
- **Visualisasi:** Loss/accuracy curves, confusion matrices, bar chart, radar chart, error analysis, trade-off plots (`figures/`).
- **Telegram Bot:** Bot real-time prediksi fungsi protein (`telegram_bot.py`) + FastAPI backend (`api/`).
- **Info Edukasi:** Data edukasi 6 famili protein (`data/class_info.json`).
