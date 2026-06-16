# LSTM — Strategi Optimasi (v1 → v2 → v3 → v4 → v5)

## Iterasi

### v1 — Baseline
| Metrik | Nilai |
|--------|-------|
| **Accuracy** | 87.0% |
| **MCC** | 0.8498 |
| **F1 Macro** | 0.88 |
| **F1 Hydrolase** | 0.76 |
| **Best Epoch** | 12 (early stop 17) |

**Masalah:** Overfitting parah (gap train-val loss ~0.25), regularisasi minim (hanya dropout 0.3 di LSTM + 0.5 di FC).

### v2 — Over-regularized
**Perubahan:** Weight Decay (1e-4), Embedding Dropout (0.2), Recurrent Dropout 0.3→0.4, FC Dropout 0.5→0.6, Combined Pooling, Gradient Clipping, LR 0.001→0.0005.

| Metrik | v1 | v2 | Delta |
|--------|:--:|:--:|:-----:|
| Accuracy | 87.0% | **86.0%** | -1.0% |
| MCC | 0.8498 | **0.8287** | -0.021 |
| F1 Hydrolase | 0.76 | **0.70** | -0.06 |
| Gap Train-Val | ~0.25 | **~0.027** | ✅ -0.223 |

✅ Overfitting hampir hilang. ❌ Tapi regularisasi terlalu agresif — kapasitas model terhambat, Hydrolase turun signifikan.

### v3 — Kurangi Regularisasi
**Perubahan:** FC Dropout 0.6→0.5, Weight Decay 1e-4→5e-5. LR 0.0005 tetap.

| Metrik | v2 | v3 | Delta |
|--------|:--:|:--:|:-----:|
| Accuracy | 86.0% | **86.7%** | +0.7% |
| MCC | 0.8287 | **0.8398** | +0.011 |
| F1 Hydrolase | 0.70 | **0.73** | +0.03 |
| Gap Train-Val | ~0.027 | **~0.079** | masih terkendali |

✅ Keseimbangan lebih baik, tren positif. ❌ Hydrolase (0.73) masih di bawah target 0.76.

### v4 — Focal Loss + Attention + Weighted Sampling ❌ Gagal
**Perubahan:** Focal Loss (γ=2.0), WeightedRandomSampler (Hydrolase weight 2×), MultiheadAttention (4 heads), Batch Size 64→32.

| Metrik | v3 | v4 | Delta |
|--------|:--:|:--:|:-----:|
| Accuracy | 86.7% | **83.0%** | ❌ -3.7% |
| MCC | 0.8398 | **0.8016** | ❌ -0.038 |
| F1 Hydrolase | 0.73 | **0.68** | ❌ -0.05 |
| Waktu/Epoch | ~56s | **~230s** | ~4× lebih lambat |

**Penyebab kegagalan:**
1. **Attention mechanism** — menambah parameter signifikan (matriks [32×4×1000×1000]), gap train-val melebar 0.079→0.171
2. **Focal Loss γ=2.0** — terlalu agresif, model "lupa" kelas yang sebelumnya sudah bagus (Oxidoreductase F1 0.82→0.74)
3. **WeightedRandomSampler** — oversampling dengan replacement menyebabkan overfit ke sampel Hydrolase spesifik (precision turun 0.72→0.60)
4. **Batch size turun** 64→32 → gradien lebih noisy
5. **Efek sinergis negatif** — kombinasi semua strategi menciptakan over-regularisasi kumulatif

### v5 — Optimasi Maksimal ✅ Sukses
**Perubahan dari v4:**
- ❌ Attention + WeightedRandomSampler — **dihapus** (penyebab utama kegagalan)
- ⚡ Focal Loss γ=2.0 → **γ=1.0** (lebih gentle)
- ✨ Label Smoothing ε=0.1 (kalibrasi probabilitas)
- ✨ Embedding 64 → **128** (kapasitas representasi 2×)
- ✨ FC 512→128→6 → **512→256→128→6** + dropout bertahap (0.5 + 0.3)
- ✨ Weight Init: Xavier (LSTM ih) + Orthogonal (LSTM hh) + Kaiming (FC)
- ⚡ Batch Size 32 → **64** (VRAM bebas setelah attention dihapus)

| Metrik | v3 | v4 | **v5** | Delta v3→v5 |
|--------|:--:|:--:|:------:|:-----------:|
| **Accuracy** | 86.7% | 83.0% | **89.0%** | ✅ +2.3% |
| **MCC** | 0.8398 | 0.8016 | **0.8710** | ✅ +0.031 |
| **F1 Macro** | 0.87 | 0.84 | **0.90** | ✅ +0.03 |
| **F1 Hydrolase** | 0.73 | 0.68 | **0.80** | ✅ +0.07 |
| **F1 Oxidoreductase** | 0.82 | 0.74 | **0.86** | ✅ +0.04 |
| **Waktu/Epoch** | ~56s | ~230s | ~283s | lebih lambat (embedding 128) |

**Semua target tercapai:** Accuracy ≥87% ✅, F1 Macro ≥0.88 ✅, MCC ≥0.85 ✅, F1 Hydrolase ≥0.76 ✅.

### Dampak Split Change (80:20 → 70:15:15)
Setelah split change, performa v5 turun:
- Accuracy: 89.0% → **86.85%**
- MCC: 0.871 → **0.842**
- F1 Hydrolase: 0.80 → **0.75**

Penyebab: training set berkurang 12.5% (20,073→17,564), kapasitas model tetap.

## Rencana Selanjutnya (v6)

### Prioritas 1: Ubah Rasio Split
Kembali ke 80:10:10 (train 20,073) untuk mengembalikan data training yang hilang.

### Prioritas 2: Optimasi Lanjutan
- Cosine annealing LR scheduler
- Uji varians seed (42, 123, 456) untuk klarifikasi baseline
- Data augmentation konservatif (mutasi 1 asam amino per sekuens dari kelompok homolog)

### Prioritas 3: Eksperimental
- Temporal Convolutional Network (TCN) sebagai alternatif arsitektur
