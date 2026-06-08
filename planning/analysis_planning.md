# Comparative Analysis (Phase 5)

## Iterasi

Analysis dilakukan dalam satu fase penuh — tidak ada iterasi (analisis komparatif adalah fase final setelah ketiga model selesai dilatih).

### Tantangan Utama
Per-sample predictions (`y_pred`, `y_true`, probabilities) **tidak disimpan** di file JSON masing-masing model — hanya metrik agregat. Notebook perlu melakukan **re-run inference** dengan memuat ulang weight ke-3 model.

### Struktur Notebook
1. Setup & Load Data — test set (3,764 sekuen) + label mapping + saved metrics dari 3 JSON
2. Load Model & Inference — CNN (`cnn_model.pth`, 2.5s), LSTM (`lstm_model_best.pth`, 4.1s), ESM-2 (`esm2_model_best/`, 313.2s)
3. Output: `y_pred`, `y_prob`, `y_true` disimpan ke `data/results/comparative_predictions.npz`

## Hasil

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

### Waktu Inference (3,764 sekuen, RTX 2050)

| Model | Waktu |
|-------|:-----:|
| CNN v2 | **2.5s** |
| LSTM v5 | 4.1s |
| ESM-2 v1 | 313.2s (~5.2 menit) |

## Catatan & Temuan

1. **ESM-2 inference lambat di RTX 2050 4GB** (313.2s) — menggunakan batch size 8 dan FP16. Loading model membutuhkan beberapa menit karena checkpoint ~140MB.
2. **Hydrolase adalah kelas tersulit** untuk semua model (CNN F1=0.61, LSTM F1=0.75, ESM-2 F1=0.87) — diversitas sekuens tertinggi.
3. **Performa inference ESM-2 sangat konsisten** dengan saved metrics: selisih hanya 0.02%.
4. **CNN vs LSTM gap lebih kecil** dari yang diantisipasi — LSTM hanya unggul ~4% accuracy.
5. **Ketidakonsistenan loading weight:** CNN dimuat dari `cnn_model.pth` (final epoch), LSTM dari `lstm_model_best.pth` (best epoch) — dapat memengaruhi perbandingan.

## Rencana Selanjutnya

- Ensemble voting (CNN + LSTM + ESM-2) untuk memanfaatkan agreement 3 model
- Analisis error lebih dalam pada hard samples (368 sampel)
- Threshold tuning per kelas untuk meningkatkan F1 Hydrolase
