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
- **Pendekatan:** Multi-scale 1D-CNN dengan 3 branch konvolusi (kernel 3, 5, dan 9) untuk menangkap motif protein dengan panjang berbeda.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Spatial Dropout (p=0.3) setelah embedding, Batch Normalization, Weight Decay.
- **Parameter:** ~340rb.
- **Training:** Lokal (RTX 2050 4GB), ~24s/epoch.
- **Inference:** 2.5s (3.764 sekuen test).
- **Hasil:** Accuracy 82.76%, F1 Macro 0.8349, MCC 0.7932.
- **Catatan:** Tidak ada versi v1; langsung dikembangkan sebagai multi-scale CNN.

### 2. LSTM v5 (Bidirectional Long Short-Term Memory)
- **Pendekatan:** Dua layer LSTM bidirectional dengan hidden size 128, menangkap long-range dependencies dalam sekuens.
- **Pooling:** Combined Global Max + Average Pooling.
- **Regularisasi:** Embedding Dropout (p=0.2), Recurrent Dropout (p=0.4), FC Dropout (p=0.5 + 0.3), Weight Decay (5e-5), Gradient Clipping.
- **Parameter:** ~827rb.
- **Training:** Lokal (RTX 2050 4GB), ~283s/epoch.
- **Inference:** 4.1s (3.764 sekuen test).
- **Optimasi:** Melalui 5 iterasi (v1→v5) — Focal Loss γ=1.0, Label Smoothing ε=0.1, Embedding 128, Xavier/Orthogonal/Kaiming weight init.
- **Hasil:** Accuracy 86.85%, F1 Macro 0.8716, MCC 0.8422.
- **Catatan:** Hasil aktual pada data split 70:15:15.

### 3. ESM-2 v1 (Evolutionary Scale Modeling)
- **Pendekatan:** LoRA fine-tuning model pretrained `esm2_t12_35M_UR50D` (35M parameter) dari Meta AI pada query/key/value projections (r=8, alpha=32, dropout=0.1).
- **Classifier Head:** MLP 480→256→6 dengan CustomClassifier.
- **Parameter:** ~34,3M total (124,7rb trainable).
- **Training:** Google Colab T4 (15.6 GB VRAM) — RTX 2050 4GB terbukti terlalu lambat (~30 menit/epoch).
- **Hyperparameter:** batch=16, accum=2, effective batch=32, FP16, AdamW (lr=1e-4).
- **Inference:** 313.2s (~5.2 menit) di RTX 2050 dengan FP16 + batch size 8.
- **Hasil:** Accuracy **93.09%**, F1 Macro **0.9327**, MCC **0.9171**.
- **Best Epoch:** 4 (dari total 9 epoch, early stopping patience=5).
- **Performa per kelas:** GPCR (0.98), Ion Channel (0.96), Kinase (0.94), Oxidoreductase (0.93), Transcription Factor (0.91), Hydrolase (0.87).

## Hasil Final (Comparative Analysis)

Perbandingan metrik pada **test set** (3.764 sekuen) dari hasil inference notebook `06_comparative_analysis.ipynb`:

| Metrik | CNN v2 | LSTM v5 | ESM-2 v1 |
|--------|:------:|:-------:|:---------:|
| **Accuracy** | 82.76% | 86.85% | **93.09%** |
| **F1 Macro** | 0.8349 | 0.8716 | **0.9327** |
| **MCC** | 0.7932 | 0.8422 | **0.9171** |
| **Precision Macro** | 0.8401 | 0.8726 | **0.9345** |
| **Recall Macro** | 0.8322 | 0.8720 | **0.9318** |
| **Parameter** | ~340rb | ~827rb | ~34,3M (124,7rb trainable) |
| **Training Device** | RTX 2050 4GB | RTX 2050 4GB | **Colab T4 16GB** |
| **Training Time** | ~2 menit | ~4 menit | ~2.5 jam |
| **Inference Time** | **2.5s** | **4.1s** | 313.2s |

**Model Agreement** (dari 3.764 sampel):
- Semua model benar: 2.868 (76.2%)
- Dua model benar: 528 (14.0%)
- Satu model benar: 228 (6.1%)
- Tidak ada yang benar: 140 (3.7%)

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
- Setiap model dilatih dengan 50 epoch, early stopping (patience=5), dan menyimpan checkpoint terbaik.
- Evaluasi: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix, Training History plots.

### Phase 5: Comparative Analysis (`06_comparative_analysis.ipynb`)
- Load model weights CNN + LSTM + ESM-2.
- Perbandingan metrik performa antar model individual.
- Visualisasi perbandingan (bar charts, confusion matrices side-by-side, radar chart).
- Analisis error dan model agreement.
- Analisis trade-off performa vs komputasi.
- Kesimpulan dan rekomendasi arsitektur terbaik.
- Menyimpan prediksi ke `data/results/comparative_predictions.npz`.

## Output

- **Notebook Jupyter (.ipynb):** Untuk setiap fase pengerjaan (data acquisition, preprocessing, 3 model, dan comparative analysis).
- **Dataset:** Data mentah (`data/raw/`) dan data siap-pakai (`data/processed/`).
- **Model Checkpoints:** Bobot model terbaik untuk CNN, LSTM, dan ESM-2 (`models/`).
- **Metrik Evaluasi:** File JSON berisi accuracy, F1-score, MCC, dan history pelatihan (`data/results/`).
- **Prediksi:** File NPZ berisi prediksi + probabilitas untuk 3 model (`data/results/comparative_predictions.npz`).
- **Visualisasi:**
  - Kurva loss dan accuracy per model (train vs validation).
  - Confusion matrix untuk setiap model.
  - Bar chart perbandingan metrik antar model.
  - Radar chart multi-metrik.
  - Error analysis visualizations.
  - Trade-off scatter plots.
- **Laporan Komparatif:**
  - Tabel perbandingan performa (accuracy, F1 macro, MCC, training time, inference time).
  - Analisis trade-off performa vs komputasi.
  - Matriks agreement antar model.
  - Analisis kelebihan dan kekurangan masing-masing arsitektur dalam konteks klasifikasi protein.

## Manfaat

- **Referensi pemilihan arsitektur:** Memberikan panduan kuantitatif bagi peneliti bioinformatika dalam memilih arsitektur deep learning yang tepat berdasarkan trade-off akurasi vs sumber daya komputasi.
- **Validasi ESM-2 untuk dataset kecil:** Menguji apakah model pretrained berskala besar (ESM-2) memberikan peningkatan signifikan dibanding arsitektur yang lebih sederhana pada dataset protein dengan jumlah terbatas.
- **Optimasi hardware menengah:** Menunjukkan strategi optimasi (gradient clipping, mixed precision, batch size tuning) yang efektif untuk GPU dengan VRAM terbatas (4GB).
- **Dasar pengembangan sistem klasifikasi protein:** Hasil perbandingan dapat digunakan sebagai dasar pengembangan pipeline klasifikasi fungsi protein otomatis untuk aplikasi penemuan obat dan annotasi genom.
- **Reproducible research:** Seluruh kode, konfigurasi, dan dataset terdokumentasi dengan baik untuk memudahkan reproduksi dan pengembangan lebih lanjut oleh peneliti lain.

## Integrasi Telegram Bot (Rencana)

Project ini memiliki rencana untuk diintegrasikan dengan **Telegram Bot** untuk memudahkan akses prediksi fungsi protein secara real-time tanpa perlu menjalankan notebook. **Fitur ini belum diimplementasikan.**

### Rencana Arsitektur

```
User → Telegram Bot → FastAPI Backend → Model (.pth)
```

### Rencana Fitur Bot

| Fitur | Deskripsi |
|-------|-----------|
| `/start` | Sambutan + daftar 6 famili protein |
| `/help` | Panduan penggunaan bot |
| `/about` | Info project & akurasi tiap model |
| `/compare <sequence>` | Prediksi dari CNN + LSTM + ESM-2 |
| `/clear` | Hapus histori percakapan |
| `/model <cnn/lstm/esm2>` | Ganti model default |
| **Input sequence** | Kirim sekuens protein → prediksi class + info edukasi |

### Rencana Output Edukatif

Setiap prediksi direncanakan menampilkan tidak hanya class, tetapi juga:
- **Deskripsi** famili protein
- **Molecular Function** (Gene Ontology)
- **Biological Process** yang terlibat
- **Cellular Location**
- **Organisme umum** yang memilikinya
- **Manfaat & relevansi** medis/industri
- **Contoh protein** dari dataset

### Detail Rancangan

Lihat [`planning/rancangan_telegram.md`](planning/rancangan_telegram.md) untuk dokumentasi lengkap.
