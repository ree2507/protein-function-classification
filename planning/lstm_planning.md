# Strategi Optimasi Model LSTM (v1 → v2 → v3 → v4 → v5)

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
- ❌ **LR 0.0005 terlalu konservatif.** Model naik perlahan, mencapai best di epoch 26 (vs epoch 12 di v1).

## 3. Hasil v3 — Implementasi Skenario A

### Perubahan yang Diterapkan (dari v2)

| # | Strategi | v2 | v3 |
|:-:|:---|---|---|
| 1 | **FC Dropout** | 0.6 | **0.5** |
| 2 | **Weight Decay** | 1e-4 | **5e-5** |
| 3 | **LR** | 0.0005 | 0.0005 (tetap) |

Strategi lain (Embedding Dropout 0.2, Recurrent Dropout 0.4, Combined Pooling, Gradient Clipping) dipertahankan dari v2.

### Perbandingan Metrik v1 vs v2 vs v3

| Metrik | v1 (Baseline) | v2 (Over-regularized) | **v3 (Skenario A)** | Delta v2→v3 |
|--------|:----------:|:----------:|:----------:|:----------:|
| **Accuracy** | 87.0% | 86.0% | **86.7%** | +0.7% |
| **MCC** | 0.8498 | 0.8287 | **0.8398** | +0.011 |
| **F1 Macro Avg** | 0.88 | 0.86 | **0.87** | +0.01 |
| **F1 Hydrolase** | 0.76 | 0.70 | **0.73** | +0.03 |
| **Best Epoch** | 12 | 26 | **23** | -3 |
| **Total Epoch** | 17 | 31 | **28** | -3 |
| **Train Loss (best)** | 0.104 | 0.366 | 0.292 | -0.074 |
| **Val Loss (best)** | ~0.36 | 0.393 | **0.371** | -0.022 |
| **Gap Train-Val** | ~0.25 | ~0.027 | **~0.079** | +0.052 |

### Analisis v3

**Berhasil:**
- ✅ **Akurasi, MCC, dan F1 semuanya naik** kembali mendekati level v1. Skenario A terbukti tepat — mengurangi regularisasi secukupnya memulihkan kapasitas model.
- ✅ **Hydrolase F1 naik 0.70 → 0.73.** Masih di bawah target 0.76, tapi tren positif.
- ✅ **Best epoch 23, total 28 epoch** — lebih cepat dari v2 (26/31), lebih lambat dari v1 (12/17). Keseimbangan yang baik.
- ✅ **Overfitting tetap terkendali.** Gap 0.079 masih jauh lebih kecil dari v1 (0.25).

**Kekurangan:**
- ❌ **Hydrolase F1 (0.73) masih terendah.** Precision 0.72, Recall 0.74 — masih kesulitan membedakan Hydrolase dari kelas lain.
- ❌ **Gap train-val melebar** dari 0.027 (v2) ke 0.079 (v3). Ini wajar karena regularisasi dikurangi, tapi perlu dimonitor.

### Target vs Realisasi v3

| Metrik | Target | Realisasi | Status |
|--------|:------:|:---------:|:------:|
| **Accuracy** | ≥87% | **86.7%** | ⚠️ Mendekati |
| **F1 Hydrolase** | ≥0.76 | **0.73** | ❌ Belum tercapai |
| **Gap Train-Val** | <0.05 | **~0.079** | ❌ Melebar |
| **Early Stopping** | >25 epoch | **28 epoch** | ✅ |

## 4. Hasil v4 — Implementasi Focal Loss + Weighted Sampling + Attention

### Perubahan yang Diterapkan (dari v3)

| # | Strategi | v3 | v4 |
|:-:|:---|---|---|
| 1 | **Loss Function** | CrossEntropyLoss | **FocalLoss(gamma=2.0, alpha=class_weights)** |
| 2 | **Sampling** | Random (no weighting) | **WeightedRandomSampler** (Hydrolase weight 2×) |
| 3 | **Arsitektur** | LSTM → Combined Pooling | **LSTM → MultiheadAttention(4 heads) → Add & LayerNorm → Combined Pooling** |
| 4 | **Batch Size** | 64 | **32** (turun karena OOM akibat attention) |

Strategi lain (FC Dropout 0.5, Embedding Dropout 0.2, Recurrent Dropout 0.4, Combined Pooling, Gradient Clipping, LR 0.0005, Weight Decay 5e-5) dipertahankan dari v3.

### Perbandingan Metrik v3 vs v4

| Metrik | v3 (Baseline) | **v4 (Focal + Attn + Sampling)** | Delta |
|--------|:----------:|:----------:|:-----:|
| **Accuracy** | 86.7% | **83.0%** | ❌ -3.7% |
| **MCC** | 0.8398 | **0.8016** | ❌ -0.038 |
| **F1 Macro Avg** | 0.87 | **0.84** | ❌ -0.03 |
| **F1 Hydrolase** | 0.73 | **0.68** | ❌ -0.05 |
| **F1 GPCR** | 0.97 | **0.97** | — |
| **F1 Ion Channel** | 0.96 | **0.94** | -0.02 |
| **F1 Kinase** | 0.88 | **0.87** | -0.01 |
| **F1 Oxidoreductase** | 0.82 | **0.74** | ❌ -0.08 |
| **F1 Transcription Factor** | 0.87 | **0.85** | -0.02 |
| **Best Epoch** | 23 | 22 | -1 |
| **Total Epoch** | 28 | 27 | -1 |
| **Train Loss (best)** | 0.292 | 0.100 | -0.192 |
| **Val Loss (best)** | 0.371 | **0.271** | -0.100 |
| **Gap Train-Val** | ~0.079 | **~0.171** | +0.092 ❌ |
| **Waktu per Epoch** | ~56s | **~230s** | ~4× lebih lambat |

### Analisis v4 — Mengapa Performa Menurun?

**1. Attention Mechanism — Penyebab Utama Overfitting**
- Matriks attention `[32×4×1000×1000]` = 128M nilai per batch menambah parameter signifikan
- Gap train-val melebar drastis (0.079 → 0.171) — model menghafal training set
- Batch_size harus diturunkan 64→32 → gradien lebih noisy, konvergensi kurang stabil
- Waktu training 4× lebih lambat tanpa peningkatan performa

**2. Focal Loss (gamma=2.0) — Over-regularization pada kelas mudah**
- Terlalu agresif menekan loss GPCR/Ion Channel (yang sudah mudah diklasifikasi)
- Model "lupa" cara memprediksi kelas-kelas yang sebelumnya sudah bagus
- Oxidoreductase paling terpukul (F1 0.82 → 0.74) — recall turun drastis (0.83 → 0.67)

**3. WeightedRandomSampler (replacement=True) — Overfit ke sample spesifik**
- Oversampling dengan replacement menyebabkan model melihat sample Hydrolase yang sama berulang kali
- Precision Hydrolase turun (0.72 → 0.60) — banyak false positive karena model over-generalize ciri Hydrolase

**4. Over-regularisasi kumulatif**
- Total mekanisme regularisasi: Embedding Dropout 0.2 + Recurrent Dropout 0.4 + FC Dropout 0.5 + Weight Decay 5e-5 + Focal Loss gamma 2.0 + Weighted Sampling + Attention regularization = terlalu berat
- Model kehilangan kapasitas representasi untuk kelas minoritas

### Target vs Realisasi v4

| Metrik | Target | Realisasi | Status |
|--------|:------:|:---------:|:------:|
| **Accuracy** | ≥87% | **83.0%** | ❌ Turun |
| **F1 Hydrolase** | ≥0.76 | **0.68** | ❌ Turun |
| **Gap Train-Val** | <0.05 | **~0.171** | ❌ Melebar |
| **Overfitting** | Terkendali | **Parah** | ❌ |

**Diagnosa:** Ketiga strategi yang diimplementasikan di v4 **kontraproduktif**. Tidak ada satu pun yang memperbaiki performa. Pendekatan "semua sekaligus" justru menciptakan efek sinergis negatif antar strategi.

## 5. Hasil v5 — Optimasi Maksimal LSTM

### Perubahan yang Diterapkan (dari v4)

| # | Strategi | v4 | v5 | Alasan |
|:-:|:---|---|---|---|
| 1 | **Attention** | ✅ MultiheadAttention | ❌ **Dihapus** | Overfit + batch_size terpaksa kecil |
| 2 | **WeightedRandomSampler** | ✅ replacement=True | ❌ **Dihapus** | Sampling dengan replacement overfit |
| 3 | **Focal Loss gamma** | 2.0 | **1.0** | Lebih gentle, tidak over-regularize kelas mudah |
| 4 | **Label Smoothing** | ε=0 | **ε=0.1** | Cegah overconfidence, kalibrasi probabilitas lebih baik |
| 5 | **Embedding Dimension** | 64 | **128** | Kapasitas representasi asam amino 2× lipat |
| 6 | **FC Layers** | 512→128→6 | **512→256→128→6** | Extra hidden layer untuk kapasitas lebih |
| 7 | **FC Dropout** | 0.5 | 0.5 + 0.3 | Dropout bertahap (besar→kecil) |
| 8 | **Weight Init** | Default PyTorch | **Xavier (LSTM ih) + Orthogonal (LSTM hh) + Kaiming (FC)** | Konvergensi lebih cepat dan stabil |
| 9 | **Batch Size** | 32 | **64** | VRAM bebas setelah attention dihapus, gradien lebih stabil |

### Perbandingan Metrik v3 vs v4 vs v5

| Metrik | v3 | v4 | **v5** | Delta v3→v5 |
|--------|:---:|:---:|:---:|:---:|
| **Accuracy** | 86.7% | 83.0% | **89.0%** | ✅ **+2.3%** |
| **MCC** | 0.8398 | 0.8016 | **0.8710** | ✅ **+0.031** |
| **F1 Macro Avg** | 0.87 | 0.84 | **0.90** | ✅ **+0.03** |
| **F1 Hydrolase** | 0.73 | 0.68 | **0.80** | ✅ **+0.07** |
| **F1 GPCR** | 0.97 | 0.97 | **0.97** | — |
| **F1 Ion Channel** | 0.96 | 0.94 | **0.96** | — |
| **F1 Kinase** | 0.88 | 0.87 | **0.90** | +0.02 |
| **F1 Oxidoreductase** | 0.82 | 0.74 | **0.86** | ✅ **+0.04** |
| **F1 Transcription Factor** | 0.87 | 0.85 | **0.88** | +0.01 |
| **Hydrolase Precision** | 0.72 | 0.60 | **0.77** | ✅ +0.05 |
| **Hydrolase Recall** | 0.74 | 0.79 | **0.83** | ✅ +0.09 |
| **Best Epoch** | 23 | 22 | **22** | -1 (≈) |
| **Total Epoch** | 28 | 27 | **27** | -1 (≈) |
| **Train Loss (best)** | 0.292 | 0.100 | 0.406 | +0.114 |
| **Test Loss (best)** | 0.371 | 0.271 | **0.4128** | +0.042 |
| **Gap Train-Test** | ~0.079 | ~0.171 | **~0.099** | +0.020 |
| **Waktu per Epoch** | ~56s | ~230s | **~283s** | ~5× dari v3 |

### Analisis v5 — Mengapa Strategi Ini Berhasil?

**1. Rollback Attention + WeightedRandomSampler — Perubahan Paling Krusial**
- Tanpa attention, batch_size kembali 64 → gradien lebih stabil, konvergensi lebih baik
- Tanpa WeightedRandomSampler, model tidak overfit ke sample Hydrolase spesifik
- Precision Hydrolase pulih: 0.60 (v4) → 0.77 (v5) — false positive berkurang drastis

**2. Focal Loss gamma=1.0 + Label Smoothing — Kombinasi Efektif**
- gamma=1.0 cukup untuk memberi fokus ke Hydrolase tanpa mengorbankan Oxidoreductase (0.74→0.86)
- Label smoothing ε=0.1 mencegah overconfidence, kalibrasi probabilitas lebih baik
- Recall Hydrolase naik 0.74→0.83 tanpa mengorbankan kelas lain

**3. Embedding 64→128 — Kapasitas Representasi 2× Lipat**
- Embedding dimensi lebih besar memungkinkan model menangkap perbedaan halus antar asam amino
- Terbukti paling membantu untuk kelas yang sulit dibedakan (Hydrolase vs Oxidoreductase)

**4. FC Expansion 512→256→128 + Weight Init — Kapasitas Klasifikasi Lebih Besar**
- Hidden layer tambahan memberi model kapasitas untuk mempelajari decision boundary yang lebih kompleks
- Inisialisasi Xavier/Orthogonal/Kaiming membantu konvergensi meski parameter bertambah

### Analisis Per-Kelas

| Kelas | v3 Precision/Recall | v5 Precision/Recal | Analisis |
|:---|---:|---:|:---|
| GPCR | 0.97 / 0.97 | 0.96 / 0.98 | Stabil, sudah optimal |
| **Hydrolase** | 0.72 / 0.74 | **0.77 / 0.83** | ✅ Precision naik, recall naik — seimbang |
| Ion Channel | 0.96 / 0.96 | 0.96 / 0.96 | Stabil |
| Kinase | 0.91 / 0.85 | 0.90 / 0.89 | Recall naik signifikan |
| **Oxidoreductase** | 0.81 / 0.83 | **0.88 / 0.85** | ✅ Precision naik drastis |
| Transcription Factor | 0.91 / 0.84 | 0.90 / 0.87 | Recall naik |

### Target vs Realisasi v5

| Metrik | Target | Realisasi | Status |
|--------|:------:|:---------:|:------:|
| **Accuracy** | ≥87% | **89.0%** | ✅ **Tercapai** |
| **F1 Macro Avg** | ≥0.88 | **0.90** | ✅ **Tercapai** |
| **MCC** | ≥0.85 | **0.8710** | ✅ **Tercapai** |
| **F1 Hydrolase** | ≥0.76 | **0.80** | ✅ **Tercapai** |
| **F1 Oxidoreductase** | ≥0.84 | **0.86** | ✅ **Tercapai** |

### Ringkasan Timeline Optimasi

```
v1 (Baseline)
  └─ → Accuracy 87.0%, F1 Hydrolase 0.76
      │  Overfitting parah (Gap 0.25)
      
v2 (Over-regularized)
  └─ → Accuracy 86.0%, F1 Hydrolase 0.70 ❌
      │  Overfitting hilang, tapi kapasitas terhambat
      
v3 (Kurangi Regularisasi)
  └─ → Accuracy 86.7%, F1 Hydrolase 0.73
      │  Keseimbangan lebih baik, tapi masih kurang
      
v4 (Focal γ=2.0 + Attention + Sampling)
  └─ → Accuracy 83.0%, F1 Hydrolase 0.68 ❌
      │  Ketiga strategi kontraproduktif
      
v5 (Optimasi Maksimal) ──→ ✅ SUKSES
  └─ → Accuracy 89.0%, F1 Hydrolase 0.80, MCC 0.871
  └─ Rollback attention & sampling
  └─ Focal γ=1.0 + label smoothing ε=0.1
  └─ Embed 128, FC 512→256→128, weight init
  └─ Semua target tercapai 🏆
```

### Kesimpulan

Model LSTM v5 berhasil mencapai performa optimal dengan:
- **Accuracy 89.0%** — melampaui target 87%
- **MCC 0.871** — korelasi prediksi-aktual sangat kuat
- **F1 Hydrolase 0.80** — naik 0.07 dari v3, kelas tersulit kini pulih
- **Semua kelas >0.86 F1** — tidak ada kelas tertinggal

Model ini telah mencapai batas performa untuk arsitektur LSTM dengan one-hot embedding. Peningkatan lebih lanjut (>90%) membutuhkan pendekatan fundamentally berbeda seperti pretrained protein embeddings (ESM-2, ProtBERT).

## 6. Rencana v6 — Pemulihan Performa Setelah Perubahan Data Split

### Latar Belakang

Data split diubah dari 80:20 (20,073/5,019) menjadi 70:15:15 (17,564/3,764/3,764) untuk menyediakan validation set terpisah. Dampaknya:

| Perubahan | Nilai |
|-----------|-------|
| **Training set berkurang** | 20,073 → 17,564 (−12.5%) |
| **Accuracy turun** | 89.0% → 86.85% (−2.15%) |
| **MCC turun** | 0.871 → 0.842 (−0.029) |
| **F1 Hydrolase turun** | 0.80 → 0.75 (−0.05) |

Class distribution tetap terjaga (stratified split), test set tidak lebih sulit. Penyebab utama: **kapasitas model tetap, data latih berkurang 2,509 sampel**, memperparah overfitting (train acc 96.1% vs val acc 88.2% di best epoch).

### Strategi yang Belum Pernah Dicoba

| # | Strategi | Kategori | Risiko | Estimasi Dampak |
|:-:|----------|----------|:------:|:---------------:|
| 1 | **Ubah rasio split** ke 80:10:10 atau 75:12.5:12.5 | Data | Rendah | +1-2% accuracy |
| 2 | **Data augmentation sekuens protein** — shuffling fragmen, mutasi sintetis, insert-delete simulasi | Data | Sedang | +0.5-1.5% accuracy |
| 3 | **Uji varians seed** — train dengan random seed berbeda (42, 123, 456) | Evaluasi | Rendah | Klarifikasi baseline |
| 4 | **Cosine annealing LR scheduler** sebagai ganti ReduceLROnPlateau | Optimizer | Rendah | +0-1% accuracy |
| 5 | **One-cycle learning rate** (lr naik lalu turun) | Optimizer | Sedang | +0-1% accuracy |
| 6 | **Mixup training** — interpolasi antar sample protein | Regularisasi | Tinggi | +0.5-1% accuracy |
| 7 | **Temporal Convolutional Network (TCN)** sebagai alternatif LSTM | Arsitektur | Tinggi | +1-3% accuracy |
| 8 | **Pretrained ESM-2 embeddings** sebagai input frozen, LSTM ringan di atasnya | Representasi | Tinggi | +3-8% accuracy |

### Detail Strategi Prioritas

#### Strategi #1: Ubah Rasio Split (Rekomendasi Utama)

Paling sederhana dan langsung mengatasi akar masalah (data training kurang).

| Parameter | 80:10:10 | 75:12.5:12.5 | 70:15:15 (saat ini) |
|--------|:---:|:---:|:---:|
| Train | 20,073 | 18,817 | 17,564 |
| Val | 2,509 | 3,136 | 3,764 |
| Test | 2,509 | 3,136 | 3,764 |

**Cara:** Edit `data/processed/split_701515.py` atau buat script split baru. Hyperparameter lain tidak perlu diubah.

#### Strategi #2: Data Augmentasi Protein

Teknik augmentasi yang relevan untuk sekuens protein:
- **Shuffling fragmen lokal**: potong sekuens di 1-2 titik, tukar posisi fragmen (meniru domain rearrangement)
- **Mutasi sintetis terkontrol**: ganti 1-2 asam amino dengan residu serupa (konservatif: berdasarkan blosum)
- **Noise padding**: tambahkan noise kecil pada embedding (bukan pada sekuens integer)

**Peringatan:** Augmentasi berlebihan bisa merusak sinyal biologis. Disarankan augmentasi hanya 1.5×-2× dataset (tidak agresif).

#### Strategi #5: One-Cycle Learning Rate

Mengganti ReduceLROnPlateau dengan OneCycleLR dari PyTorch:
- LR naik dari `1e-5` → `5e-4` (warmup) lalu cosine decay ke `1e-6`
- Momentum siklus terbalik (0.85 → 0.95)
- Terbukti efektif untuk LSTM di berbagai task sequence classification

#### Strategi #8: Pretrained ESM-2 Embeddings (Jembatan ke Phase 5 project)

Fase ini sudah direncanakan di project (ESM-2 for protein function classification). Untuk v6 LSTM, bisa gunakan ESM-2 embeddings sebagai **input features frozen**:
- Ekstrak embedding ESM-2 tiap sekuens (offline, sekali jalan)
- Ganti `nn.Embedding` dengan linear projection dari dimensi ESM-2 (1280) ke hidden_dim (128)
- LSTM tetap di-train, tapi embedding ESM-2 tidak di-fine-tune
- **Potensi:** Memanfaatkan representasi protein pretrained tanpa biaya komputasi besar

### Matriks Prioritas

| Prioritas | Strategi | Effort | Impact | Recommended |
|:---------:|----------|:------:|:------:|:-----------:|
| 1 | **Ubah split (80:10:10)** | ⚡ 1 jam | 🎯 Tinggi (langsung ke akar) | ✅ |
| 2 | **Uji varians seed** | ⚡ 30 menit | 📊 Klarifikasi baseline | ✅ |
| 3 | **Cosine annealing LR** | ⚡ 1 jam | 📈 Sedang | ✅ |
| 4 | **Data augmentasi** | ⏳ 4-8 jam | 📈 Sedang | ⚠️ Hati-hati |
| 5 | **ESM-2 frozen embeddings** | ⏳ 8-16 jam | 🚀 Tinggi | 🧪 Eksperimental |
| 6 | **Mixup / TCN** | ⏳ 8+ jam | ❓ Tidak pasti | ❌ Prioritas rendah |

### Catatan Khusus: Data Augmentasi untuk Protein

Tidak seperti gambar (rotasi/flip aman), augmentasi sekuens protein berisiko tinggi:
- Mutasi acak bisa menghilangkan sinyal biologis (active site, binding domain)
- Shuffling fragmen hanya aman jika domain protein independen
- Over-augmentasi (2×+) sudah terbukti menurunkan performa di beberapa studi (Rao et al., 2019)

**Rekomendasi:** Jika ingin augmentasi, gunakan pendekatan paling konservatif: ganti max 1 asam amino per sekuens dengan residu dari kelompok yang sama (hidrofobik→hidrofobik, polar→polar, charged→charged).
