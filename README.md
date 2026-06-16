# Klasifikasi Fungsi Protein dengan LSTM

## Status Pengerjaan

Proyek fokus pada pengembangan model LSTM untuk klasifikasi fungsi protein. CNN dan ESM-2 telah dihapus dari project. Fase data acquisition, preprocessing, LSTM v5, dan integrasi Telegram Bot telah selesai. Fase pengembangan model lanjutan masih direncanakan (lihat `planning/`).

## Deskripsi Project

Project ini bertujuan untuk mengklasifikasikan fungsi protein berdasarkan sekuens asam amino menggunakan arsitektur LSTM (Long Short-Term Memory). Klasifikasi fungsi protein merupakan tugas fundamental dalam bioinformatika yang dapat mempercepat penemuan obat dan penelitian biologi.

## Masalah

- **Keterbatasan alignment-based methods:** Metode klasifikasi protein tradisional seperti BLAST bergantung pada sequence alignment yang lambat dan tidak scalable untuk dataset besar.
- **Ketidakseimbangan performa per kelas:** Beberapa kelas protein (terutama Hydrolase) secara konsisten memiliki F1-score lebih rendah dari kelas lain.

## Urgensi

Pemilihan arsitektur deep learning yang tepat untuk klasifikasi protein sangat bergantung pada trade-off antara akurasi, kecepatan, dan sumber daya komputasi. Project ini mengoptimalkan model LSTM untuk memberikan performa terbaik pada hardware lokal.

## Tujuan

- Mengembangkan model LSTM yang optimal untuk klasifikasi fungsi protein.
- Mengoptimalkan pelatihan model untuk hardware lokal.
- Mengidentifikasi strategi regularisasi dan hyperparameter tuning yang efektif.
- Meningkatkan performa klasifikasi pada kelas-kelas yang sulit (seperti Hydrolase).

## Manfaat

- **Referensi implementasi LSTM:** Memberikan panduan implementasi LSTM untuk klasifikasi sekuens protein.
- **Optimasi hardware menengah:** Menunjukkan strategi optimasi (gradient clipping, mixed precision, batch size tuning) yang efektif untuk GPU dengan VRAM terbatas (4GB).
- **Dasar pengembangan sistem klasifikasi protein:** Hasil dapat digunakan sebagai dasar pengembangan pipeline klasifikasi fungsi protein otomatis.

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

### LSTM v5 (Bidirectional Long Short-Term Memory)
- **Pendekatan:** Dua layer LSTM bidirectional dengan hidden size 128 — menangkap long-range dependencies dalam sekuens.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Embedding Dropout (p=0.2), Recurrent Dropout (p=0.4), FC Dropout (p=0.5 + 0.3), Weight Decay (5e-5), Gradient Clipping.
- **Parameter:** ~827rb.
- **Training:** Lokal (RTX 2050 4GB), ~283s/epoch.
- **Optimasi:** 5 iterasi (v1→v5) — Focal Loss γ=1.0, Label Smoothing ε=0.1, Embedding 128, Xavier/Orthogonal/Kaiming weight init.
- **Hasil:** Accuracy 86.85%, F1 Macro 0.8716, MCC 0.8422.
- Detail optimasi: [`planning/lstm_planning.md`](planning/lstm_planning.md).

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
- **LSTM (`04_lstm_model.ipynb`)** — Bidirectional LSTM v5.
- 50 epoch, early stopping (patience=5), checkpoint terbaik.
- Evaluasi: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix, Training History.

### Phase 5: Telegram Bot Integration
- **Backend API:** FastAPI (`api/main.py`) — endpoint `/predict`, `/health`, `/models`.
- **Bot Telegram:** `telegram_bot.py` — 3 command handler (`/start`, `/help`, `/about`) + auto-detect sequence.
- **Model Serving:** Lazy loading dengan caching via `api/model_loader.py`.
- **Output Edukatif:** Setiap prediksi menampilkan deskripsi famili, fungsi molekuler, proses biologis, lokasi sel, dan rincian probabilitas.
- Detail implementasi: [`planning/rancangan_telegram.md`](planning/rancangan_telegram.md).

## Output

- **Notebook Jupyter (.ipynb):** Setiap fase pengerjaan (01–04).
- **Dataset:** Data mentah (`data/raw/`) dan data siap-pakai (`data/processed/`).
- **Model Checkpoints:** Bobot model terbaik (`models/`).
- **Metrik Evaluasi:** File JSON accuracy, F1-score, MCC, history pelatihan (`data/results/`).
- **Telegram Bot:** Bot real-time prediksi fungsi protein (`telegram_bot.py`) + FastAPI backend (`api/`).
- **Info Edukasi:** Data edukasi 6 famili protein (`data/class_info.json`).
