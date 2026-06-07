# Laporan Comparative Analysis (Phase 5)

## Tujuan
Notebook `06_comparative_analysis.ipynb` yang membandingkan performa CNN v2, LSTM v5, dan ESM-2 v1 secara visual dan analitis — tanpa ensembling.

## Tantangan Utama
Per-sample predictions (`y_pred`, `y_true`, probabilities) **tidak disimpan** di file JSON — hanya metrik agregat. Notebook perlu melakukan **re-run inference** dengan memuat ulang weight model.

## Struktur Notebook

### Bagian 1: Setup & Load Data
- Import libraries
- Load test set (`data/processed/test.csv`) — 3,764 sequences
- Load label mapping (`data/processed/label_mapping.json`)
- Load saved metrics dari 3 JSON (`data/results/*_metrics.json`)

### Bagian 2: Load Model & Inference
| Model | Source Weight | Waktu Aktual |
|-------|---------------|:--------------:|
| **CNN v2** | `models/cnn_model.pth` | **2.5 detik** |
| **LSTM v5** | `models/lstm_model_best.pth` | **4.1 detik** |
| **ESM-2 v1** | `models/esm2_model_best/` (LoRA) | **313.2 detik** (~5.2 menit) |

Output yang disimpan:
- `y_pred` (predicted class), `y_prob` (probabilities), `y_true` (ground truth)
- Inference time per model
- Simpan ke `data/results/comparative_predictions.npz`

### Bagian 3: Overall Performance Comparison
- **Bar chart** side-by-side: Accuracy, F1 Macro, MCC (3 models × 3 metrics)
- **Radar chart** multi-metrik
- **Confidence histogram** — distribusi max probability per model

### Bagian 4: Per-Class Performance
- **Grouped bar chart** — F1, Precision, Recall per class untuk 3 model
- Identifikasi termudah/tersulit per model
- Sorot **Hydrolase** sebagai kelas paling menantang

### Bagian 5: Confusion Matrices
- 3 confusion matrices dalam 1 figure (subplot 1×3)
- Versi normalized untuk perbandingan yang adil

### Bagian 6: Error Analysis
- **Model Agreement Matrix** — heatmap seberapa sering model setuju/salah bersama
- **Hard Sample Identification** — sampel salah oleh 2+ model
- **Error patterns** per kelas

### Bagian 7: Training Dynamics Comparison
- Loss curves (train & val) untuk 3 model dalam 1 plot
- Accuracy curves overlay
- Analisis convergence speed & overfitting

### Bagian 8: Trade-off Analysis
- Scatter plot: Accuracy vs Parameter Count vs Training Time vs VRAM
- Tabel perbandingan resource lengkap
- Rekomendasi berdasarkan use case

### Bagian 9: Kesimpulan & Rekomendasi
- ESM-2 unggul akurasi, CNN unggul efisiensi, LSTM sebagai kompromi

## Hasil Aktual

### Metrik Final

| Metrik | CNN v2 | LSTM v5 | ESM-2 v1 |
|--------|:------:|:-------:|:---------:|
| **Accuracy** | 82.76% | 86.85% | **93.09%** |
| **F1 Macro** | 0.8349 | 0.8716 | **0.9327** |
| **MCC** | 0.7932 | 0.8422 | **0.9171** |
| **Precision Macro** | 0.8401 | 0.8726 | **0.9345** |
| **Recall Macro** | 0.8322 | 0.8720 | **0.9318** |

### F1-Score per Kelas

| Kelas | CNN v2 | LSTM v5 | ESM-2 v1 |
|-------|:------:|:-------:|:---------:|
| GPCR | 0.9698 | 0.9594 | **0.9797** |
| Hydrolase | 0.6137 | 0.7456 | **0.8738** |
| Ion Channel | 0.9500 | 0.9559 | **0.9624** |
| Kinase | 0.8428 | 0.8703 | **0.9393** |
| Oxidoreductase | 0.7889 | 0.8340 | **0.9277** |
| Transcription Factor | 0.8441 | 0.8647 | **0.9135** |

### Model Agreement (3,764 samples)

| Kategori | Jumlah | Persentase |
|----------|:------:|:----------:|
| Semua model benar | 2,868 | 76.2% |
| Dua model benar | 528 | 14.0% |
| Satu model benar | 228 | 6.1% |
| Tidak ada yang benar | 140 | 3.7% |
| Hard samples (salah ≥2 model) | 368 | 9.8% |

### Waktu Inference (3,764 sekuen)

| Model | Waktu |
|-------|:-----:|
| CNN v2 | **2.5s** |
| LSTM v5 | 4.1s |
| ESM-2 v1 | 313.2s (~5.2 menit) |

## Files Generated

| File | Deskripsi |
|------|-----------|
| `data/results/comparative_predictions.npz` | Semua prediksi + probabilitas (3 model × 3,764 samples) |
| `figures/comparison_overall.png` | Bar chart perbandingan metrik |
| `figures/comparison_perclass.png` | Per-class performance |
| `figures/confusion_matrices.png` | Side-by-side confusion matrices |
| `figures/error_analysis.png` | Error analysis visualizations |
| `figures/training_curves.png` | Overlay training curves |
| `figures/tradeoff.png` | Trade-off scatter plot |
| `figures/F1_Radar_Chart.png` | Radar chart perbandingan F1 per kelas |
| `figures/confidence_histogram.png` | Distribusi confidence per model |

## Catatan & Temuan

1. **ESM-2 inference lambat di RTX 2050 4GB** (313.2s) — menggunakan batch size 8 dan FP16. Loading model saja membutuhkan beberapa menit karena ukuran checkpoint ~140MB.
2. **Hydrolase adalah kelas tersulit** untuk semua model (CNN F1=0.61, LSTM F1=0.75, ESM-2 F1=0.87) — konsisten dengan dugaan bahwa kelas ini memiliki diversitas sekuens tertinggi.
3. **Performa inference ESM-2 sangat konsisten** dengan saved metrics: Acc 93.09% (saved) vs 93.07% (inference) — selisih hanya 0.02%.
4. **CNN vs LSTM gap lebih kecil** dari yang diantisipasi pada awalnya — LSTM hanya unggul ~4% dalam accuracy.
