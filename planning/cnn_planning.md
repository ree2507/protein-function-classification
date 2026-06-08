# CNN — Strategi Optimasi

## Iterasi

### Baseline (v1 — sebelum notebook final)
- **Akurasi:** ~83.4%, **MCC:** 0.8020
- **Masalah:** Overfitting (val loss naik setelah epoch 8), kernel size tunggal (k=5) gagal menangkap motif multi-panjang, F1 Hydrolase terendah (0.69)
- **Keputusan:** Tidak ada versi v1 di notebook; langsung dikembangkan sebagai multi-scale CNN v2

### v2 (Final — `03_cnn_model.ipynb`)
**Perubahan:**
- Multi-scale 1D-CNN: 3 branch konvolusi (kernel 3, 5, 9) + concatenation
- Combined Global Max + Average Pooling (6 × 128 = 768 fitur)
- Spatial Dropout (`nn.Dropout1d(p=0.3)`) setelah embedding
- Weight Decay (1e-4), Batch Normalization di setiap branch
- ReduceLROnPlateau (factor=0.5, patience=2), class weights balanced
- Early stopping patience=5, 50 epochs max

**Hasil vs Baseline:**

| Metrik | Baseline (v1) | v2 (80:20 split) | v2 (70:15:15 split) |
|--------|:-------------:|:-----------------:|:-------------------:|
| Akurasi | ~83.4% | ~83.5% | **82.76%** |
| MCC | 0.8020 | 0.8026 | **0.7932** |
| F1 Macro | — | 0.84 | **0.83** |
| F1 Hydrolase | 0.69 | 0.65 | **0.61** |
| Best Epoch | — | 25 | **21** |
| Waktu/Epoch | — | ~127-145s | **~24s** |

**Dampak split change (80:20 → 70:15:15):** Training set berkurang 12.5% (20,073→17,564). Hydrolase paling terpukul (F1 0.65→0.61). GPCR dan Ion Channel tetap stabil (F1 >0.95).

## Rencana Selanjutnya (v3)

### Prioritas 1: Data-Level
- Sequence augmentation (mutasi homolog, shuffling fragmen)
- Oversample Hydrolase
- Ubah rasio split ke 80:10:10 untuk mengembalikan data training

### Prioritas 2: Loss Function
- Focal Loss (γ=1.0) untuk fokus pada kelas sulit
- Label Smoothing (ε=0.1) untuk cegah overconfidence

### Prioritas 3: Hyperparameter Tuning
- Embedding dimension search (32, 64, 128, 256)
- Dropout rate tuning (spatial: 0.2/0.3/0.4, FC: 0.3/0.5/0.6)
- Learning rate & scheduler experiment (CosineAnnealingLR, OneCycleLR)

### Target v3
- Akurasi: 86%–88%
- F1 Hydrolase: >0.70
- Kurva loss lebih stabil tanpa early stopping dini
