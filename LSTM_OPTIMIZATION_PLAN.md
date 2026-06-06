# Strategi Optimasi Model LSTM (v2 → v3)

## 1. Hasil Baseline (v1) — 04_lstm_model.ipynb

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | 87.0% |
| **MCC** | 0.8498 |
| **F1 Macro Avg** | 0.88 |
| **F1 Hydrolase** | 0.76 |
| **Best Epoch** | 12 |
| **Early Stopping** | Epoch 17 (patience=5) |
| **Train Loss (best)** | 0.104 |
| **Val Loss (best)** | ~0.36 |
| **Gap Train-Val** | ~0.25 |

### Analisis Masalah Utama (v1)
1. **Overfitting:** Train loss turun drastis (1.199 → 0.104), val loss stagnan di ~0.36 setelah epoch 12. Gap sangat lebar.
2. **Hydrolase masih terendah:** F1 0.76 (lebih baik dari CNN 0.65, tapi masih di bawah kelas lain yang rata-rata >0.87).
3. **Regularisasi minim:** Tidak ada weight_decay, tidak ada spatial dropout, hanya mengandalkan dropout 0.3 di LSTM dan 0.5 di FC layer.

## 2. Hasil v2 — Setelah Optimasi

### Perubahan yang Diterapkan

| # | Strategi | Detail |
|:-:|:---|---|
| 1 | **Weight Decay** | `weight_decay=1e-4` pada Adam optimizer |
| 2 | **Embedding Dropout** | `nn.Dropout(p=0.2)` setelah Embedding layer |
| 3 | **Recurrent Dropout** | dinaikkan dari 0.3 → **0.4** |
| 4 | **FC Dropout** | dinaikkan dari 0.5 → **0.6** |
| 5 | **Combined Pooling** | Max + Average Pooling, di-concatenate (input FC: 256 → 512) |
| 6 | **Gradient Clipping** | `max_norm=1.0` setelah `loss.backward()` |
| 7 | **Learning Rate** | diturunkan dari 0.001 → **0.0005** |

### Perbandingan Metrik v1 vs v2

| Metrik | v1 (Baseline) | v2 (Optimized) | Delta |
|--------|:----------:|:----------:|:-----:|
| **Accuracy** | 87.0% | **86.0%** | -1.0% |
| **MCC** | 0.8498 | **0.8287** | -0.021 |
| **F1 Macro Avg** | 0.88 | **0.86** | -0.02 |
| **F1 Hydrolase** | 0.76 | **0.70** | -0.06 |
| **Best Epoch** | 12 | **26** | +14 |
| **Total Epoch** | 17 | 31 | +14 |
| **Train Loss (best)** | 0.104 | 0.366 | +0.262 |
| **Val Loss (best)** | ~0.36 | 0.393 | +0.033 |
| **Gap Train-Val** | ~0.25 | **~0.027** | **-0.223** |

### Analisis v2

**Berhasil:**
- ✅ **Overfitting hampir hilang.** Gap train-val loss mengecil drastis dari ~0.25 → ~0.027. Regularisasi bekerja sangat baik.
- ✅ **Training lebih stabil.** Tidak ada loss spikes berkat gradient clipping.
- ✅ **Model belajar lebih lama** (31 vs 17 epoch) tanpa overfit dini, menandakan regularisasi efektif.

**Kekurangan:**
- ❌ **Akurasi & MCC turun tipis.** Regularisasi terlalu agresif — kombinasi 4 mekanisme regularisasi (weight decay + embedding dropout + dropout 0.4 LSTM + dropout 0.6 FC) menghambat kapasitas model.
- ❌ **Hydrolase F1 turun signifikan (0.76 → 0.70).** Kelas hard-to-learn paling terdampak oleh regularisasi berlebih.
- ❌ **LR 0.0005 terlalu konservatif.** Model naik perlahan, mencapai best di epoch 26 (vs epoch 12 di v1), kemungkinan belum mencapai potensi maksimal dalam 31 epoch.

## 3. Strategi v3 — Rekomendasi

### Prioritas Implementasi v3

| Prioritas | Langkah | Dampak yang Diharapkan |
|:---:|:---|:---|
| 1 | **FC Dropdown: 0.6 → 0.5** | Mengurangi over-regularisasi, memberikan kapasitas lebih untuk kelas sulit |
| 2 | **Weight Decay: 1e-4 → 5e-5** | Regularisasi L2 lebih ringan, tetap cegah overfitting tanpa menekan performa |
| 3 | **Learning Rate: 0.0005 → 0.001** | Kembali ke LR awal, konvergensi lebih cepat (didukung regularisasi yang sudah ditambahkan) |
| 4 | **AU: LR 0.0005 + FC Dropout 0.5** | Alternatif jika LR 0.001 terlalu agresif — pertahankan LR rendah tapi kurangi dropout |
| 5 | **Attention Mechanism** | Tambahkan `nn.MultiheadAttention` atau Bahdanau attention setelah LSTM, sebelum pooling — fokus ke region sekuens paling informatif |
| 6 | **CosineAnnealingLR** | Ganti ReduceLROnPlateau dengan CosineAnnealingLR untuk scheduling yang lebih smooth |
| 7 | **Class-weighted Sampling** | Oversample kelas Hydrolase di DataLoader untuk menyeimbangkan representasi |

### Eksperimen yang Disarankan

**Skenario A (Recommended):** FC Dropout 0.6→0.5, Weight Decay 1e-4→5e-5, LR tetap 0.0005
**Skenario B:** FC Dropout 0.6→0.5, LR 0.0005→0.001, Weight Decay tetap 1e-4
**Skenario C:** Attention + Combined Pooling + Skenario A

## 4. Target Outcome (v3)
*   Akurasi kembali ke **≥87%** dengan gap train-val loss tetap kecil (<0.05).
*   F1-Score Hydrolase **≥0.76** (kembali ke level v1 atau lebih baik).
*   Early stopping >25 epoch dengan val loss yang terus menurun stabil.
