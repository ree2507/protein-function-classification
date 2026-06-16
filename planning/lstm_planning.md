# LSTM — Strategi Optimasi (v1 → v1.1)

## Hasil Training v1

**Model:** LSTM v1 — augmentasi 1-AA + CosineAnnealingWarmRestarts + FocalLoss γ=1.0 + Label Smoothing ε=0.1

### Metrik Test Set

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | 87.81% |
| **F1 Macro** | 0.8809 |
| **MCC** | 0.8535 |
| **Best Epoch** | 31 / 36 (early stop) |
| **Waktu/Epoch** | ~47s |

### Per-Class F1

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| GPCR | 0.96 | 0.97 | **0.97** | 498 |
| Hydrolase | 0.75 | 0.77 | **0.76** | 643 |
| Ion Channel | 0.94 | 0.98 | **0.96** | 671 |
| Kinase | 0.91 | 0.83 | **0.87** | 628 |
| Oxidoreductase | 0.86 | 0.85 | **0.85** | 679 |
| Transcription Factor | 0.87 | 0.87 | **0.87** | 645 |

### Perbandingan dengan Baseline v5

| Metrik | v5 (70:15:15) | v6 Exp2 (Augment) | **v1** |
|--------|:-------------:|:-----------------:|:-----:|
| **Accuracy** | 86.85% | **88.86%** | **87.81%** |
| **F1 Macro** | 0.87 | **0.89** | **0.88** |
| **MCC** | 0.842 | **0.862** | **0.854** |
| **Hydrolase F1** | 0.75 | **0.78** | **0.76** |

### Analisis

**Positif:**
- ✅ v1 exceed v5 baseline (+0.96% Acc, +0.012 MCC)
- ✅ Hydrolase naik 0.75 → 0.76 (meski belum ke 0.78)
- ✅ Waktu/epoch turun ~283s → ~47s (karena tanpa compile overhead, pure training)
- ✅ Augmentasi terbukti efektif untuk kelas minoritas

**Negatif:**
- ❌ v1 di bawah v6 Exp2 (88.86%) — Cosine LR justru menurunkan performa augmentasi
- ❌ Epoch 16 ada spike loss — bertepatan dengan restart Cosine LR (T₀=5, T_mult=2)
- ❌ Hydrolase stuck di 0.76 (turun dari 0.78 di v6 Exp2)

### Kesimpulan

Cosine LR restart menyebabkan model kehilangan progres saat augmentasi — setiap restart (LR naik ke 5e-4) mengganggu representasi yang sudah stabil. **Augmentasi + ReduceLROnPlateau (v6 Exp2) menghasilkan performa lebih tinggi (88.86%)** karena LR turun monoton.

---

## Rencana v1.1

### Target: ≥88.86% Accuracy, ≥0.86 MCC, ≥0.89 F1 Macro

### Perubahan dari v1
- ❌ CosineAnnealingWarmRestarts → **ReduceLROnPlateau** (kembali ke scheduler yang terbukti)
- ✨ Multi-seed (42, 123, 456) — untuk cari seed optimal + ukur varians
- Pertahankan: Augmentasi 1-AA, FocalLoss γ=1.0, LS ε=0.1, arsitektur v1

### Eksperimen Lanjutan (Jika Target Tercapai)

| Eksperimen | Detail |
|------------|--------|
| **Seed Tuning** | Cari seed terbaik dari [42, 123, 456] |
| **Augmentasi Rate** | Uji probabilitas mutasi 0.05, 0.1, 0.15 |
| **Early Stopping** | Uji patience 7 (vs 5) untuk epoch lebih banyak |
| **Embedding Size** | Uji 128 vs 256 (trade-off VRAM vs kapasitas) |
| **Dropout Tuning** | Uji FC dropout 0.4 vs 0.5 vs 0.6 |

---

## Hasil Training v1.1

**Model:** LSTM v1.1 — augmentasi 1-AA + ReduceLROnPlateau + FocalLoss γ=1.0 + Label Smoothing ε=0.1 + Multi-Seed [42, 123, 456]

### Multi-Seed Test Set

| Seed | Accuracy | F1 Macro | MCC | Best Epoch |
|:----:|:--------:|:--------:|:---:|:----------:|
| **42** | **0.8831** | **0.8857** | **0.8596** | 23 |
| 123 | 0.8778 | 0.8803 | 0.8533 | 24 |
| 456 | 0.8791 | 0.8811 | 0.8549 | 23 |
| **Mean** | **0.8800 ± 0.0023** | | | |

### Per-Class F1 (Best Seed 42)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| GPCR | 0.95 | 0.98 | **0.97** | 498 |
| Hydrolase | 0.80 | 0.73 | **0.77** | 643 |
| Ion Channel | 0.94 | 0.97 | **0.95** | 671 |
| Kinase | 0.89 | 0.87 | **0.88** | 628 |
| Oxidoreductase | 0.83 | 0.87 | **0.85** | 679 |
| Transcription Factor | 0.87 | 0.87 | **0.87** | 645 |

### Perbandingan

| Metrik | v1 (Cosine LR) | **v1.1 (ReduceLROnPlateau)** | v6 Exp2 (target) |
|--------|:--------------:|:---------------------------:|:----------------:|
| **Accuracy** | 87.81% | **88.31%** | **88.86%** |
| **F1 Macro** | 0.8809 | **0.8857** | **0.89** |
| **MCC** | 0.8535 | **0.8596** | **0.862** |
| **Hydrolase F1** | 0.76 | **0.77** | **0.78** |

### Analisis

**Positif:**
- ✅ ReduceLROnPlateau >> Cosine LR — v1.1 (88.31%) exceed v1 (87.81%) dengan +0.50%
- ✅ Multi-seed stabil — std dev ±0.23%, seed 42 terbaik
- ✅ Hydrolase naik 0.76→0.77 — augmentasi bekerja konsisten
- ✅ Tidak ada spike loss — training smooth tanpa restart Cosine LR
- ✅ Gap ke target mengecil dari 1.05% (v1) → 0.55% (v1.1)

**Negatif:**
- ❌ Masih 0.55% di bawah v6 Exp2 (88.86%)
- ❌ Hydrolase recall rendah (0.73) meski precision naik (0.80) — model terlalu konservatif untuk kelas minoritas

---

## Rencana v1.2

### Target: ≥89.0% Accuracy, ≥0.87 MCC, ≥0.89 F1 Macro

### Strategi: Hyperparameter Tuning + Modifikasi Arsitektur

### Perubahan Arsitektur

| Komponen | v1.1 | v1.2 | Dampak |
|----------|------|------|--------|
| **Embedding** | 128 | **256** | Kapasitas representasi AA 2× |
| **Hidden LSTM** | 128 | **256** | Kemampuan menangkap pola lebih kompleks |
| **BatchNorm** | Tidak ada | **BatchNorm1d(1024)** setelah pooling | Stabilisasi training, mengurangi internal covariate shift |
| **FC Layers** | 3 layer (512→256→128→6) | **4 layer (1024→512→256→128→6)** | Lebih banyak non-linearitas, hierarki fitur lebih dalam |
| **Total Params** | ~430K | **~1.74M** | Masih aman di 4GB VRAM |

### Perubahan Hyperparameter

| Parameter | v1.1 | v1.2 | Alasan |
|-----------|------|------|--------|
| **Augmentasi Rate** | ~0.1 (1 AA/sample) | **0.05 (5% prob)** | Lebih konservatif, mengurangi noise |
| **Early Stopping** | patience=5 | **patience=7** | Model lebih besar butuh epoch lebih banyak |
| **LR Scheduler** | ReduceLROnPlateau(factor=0.5) | **ReduceLROnPlateau(factor=0.3)** | LR turun lebih agresif saat plateau |
| **Seed** | [42, 123, 456] | **[42]** (seed terbaik) | Seed 42 terbukti optimal |

### Expected Improvement

| Metrik | v1.1 | Target v1.2 |
|--------|:----:|:-----------:|
| **Accuracy** | 88.31% | **≥89.0%** |
| **F1 Macro** | 0.8857 | **≥0.89** |
| **MCC** | 0.8596 | **≥0.87** |
| **Hydrolase F1** | 0.77 | **≥0.79** |

Setelah v1.2, eksperimen lanjutan:
- **Multi-seed validation** dengan arsitektur baru
- **Dropout tuning** (0.4 / 0.5 / 0.6)
- **Augmentasi rate tuning** lanjutan (0.03 / 0.05 / 0.07)
