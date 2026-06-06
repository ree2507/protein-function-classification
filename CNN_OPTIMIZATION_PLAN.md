# Strategi Optimasi Model CNN (Phase 4a - v2)

## 1. Analisis Baseline (v1)

*   **Akurasi:** ~83.4%
*   **MCC:** 0.8020
*   **Kekurangan Utama:**
    *   **Overfitting:** Validation loss mulai naik setelah epoch 8, sementara training loss terus turun.
    *   **Performa Kelas Rendah:** Kelas *Hydrolase* memiliki F1-Score terendah (0.69) dibandingkan kelas lain (rata-rata >0.85).
    *   **Fitur Terbatas:** Penggunaan kernel size tunggal (k=5) mungkin gagal menangkap motif protein yang lebih pendek atau lebih panjang.

## 2. Implementasi v2 (03_cnn_model.ipynb) — Status: ✅ SELESAI

### A. Arsitektur: Multi-Scale 1D-CNN ✅
3 branch konvolusi (kernel 3, 5, 9) dengan hasil digabung via concatenation.

### B. Combined Global Pooling ✅
Max + Average pooling di setiap branch, di-concatenate (6 x 128 = 768 fitur).

### C. Regularisasi ✅
- Spatial Dropout (`nn.Dropout1d(p=0.3)`) setelah embedding
- Weight Decay (L2, `1e-4`) di optimizer Adam
- Batch Normalization di setiap branch

### D. Optimasi Training ✅
- ReduceLROnPlateau (factor=0.5, patience=2)
- Class weights balanced di CrossEntropyLoss
- Early stopping patience=5, 50 epochs max

## 3. Hasil v2 vs Target

| Metrik | Target | Aktual (v2) | Status |
|--------|--------|-------------|--------|
| Akurasi | 86% - 88% | **~84%** | ❌ Belum tercapai |
| F1-Score Hydrolase | >0.75 | **0.65** | ❌ Belum tercapai |
| Stabilitas loss | Gap kecil | Gap mengecil (validasi stagnan ~epoch 25) | ⚠️ Sebagian |

**Analisis:** Semua strategi sudah diterapkan sepenuhnya, namun target belum tercapai. Hydrolase masih menjadi kelas tersulit (precision 0.59, recall 0.73). Validasi loss berhenti membaik di epoch 25 (early stopping di epoch 30).

## 4. Rencana v3 — Strategi Lanjutan

### E. Data-Level Improvements
1. **Sequence Augmentation:**
   - Reverse complement setiap protein sequence (data 2x lipat)
   - Random mutation sampling (mengubah 1-2 asam amino secara acak, dengan probabilitas kecil)
2. **Upsampling Kelas Hydrolase:**
   - Oversample data Hydrolase di training set untuk memberi lebih banyak contoh
3. **Filter Noise Sequence:**
   - Analisis sequence length distribution per kelas, buang outlier ekstrim yang mungkin noise

### F. Feature Representation
1. **Embedding Dimension Search:**
   - Uji `embed_dim` = 32, 64 (saat ini), 128, 256
   - Embedding yang lebih besar mungkin menangkap lebih banyak informasi biokimia
2. **Pretrained Amino Acid Embeddings:**
   - Gunakan ProtVec (word2vec pretrained pada sekuens protein) sebagai inisialisasi embedding layer, bukan random

### G. Arsitektur Deeper
1. **Multi-Layer Convolutions:**
   - Setiap branch: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → ReLU (bukan 1 layer saja)
2. **Residual Connections:**
   - Tambahkan skip connections agar gradient flow lebih baik di jaringan yang lebih dalam
3. **Attention Mechanism:**
   - Tambahkan attention layer setelah concatenation pooling, sebelum FC layer
   - Bisa pakai `nn.MultiheadAttention` sederhana

### H. Hyperparameter Tuning
1. **Dropout Rate:**
   - Uji spatial dropout: 0.2, 0.3 (saat ini), 0.4
   - Uji FC dropout: 0.3, 0.5 (saat ini), 0.6
2. **Learning Rate & Scheduler:**
   - Uji `lr` awal = 0.0005 atau 0.0003
   - Uji `CosineAnnealingLR` sebagai alternatif `ReduceLROnPlateau`
3. **Weight Decay:**
   - Uji `weight_decay` = 1e-4 (saat ini), 5e-4, 1e-3

### I. Loss Function Experimentation
1. **Focal Loss:**
   - Mengganti CrossEntropyLoss dengan Focal Loss untuk fokus pada kelas sulit (Hydrolase)
   - Focal Loss secara otomatis mengurangi kontribusi loss dari kelas yang sudah mudah diklasifikasikan
2. **Label Smoothing:**
   - Menambahkan label smoothing ke CrossEntropyLoss untuk mencegah overconfidence

## 5. Prioritas Implementasi v3

| Prioritas | Langkah | Dampak yang Diharapkan |
|:---:|:---|:---|
| 1 | **Sequence Augmentation** (E1) | Data 2x → model lihat lebih banyak variasi |
| 2 | **Multi-Layer Conv** (G1) | Representasi fitur lebih hierarkis |
| 3 | **Focal Loss** (I1) | Fokus ke Hydrolase, kelas minoritas |
| 4 | **Hyperparameter Tuning** (H) | Optimalisasi dropout & LR |
| 5 | **Attention** (G3) | Model bisa fokus ke region penting |
| 6 | **Pretrained Embeddings** (F2) | Transfer learning dari data protein lain |

## 6. Target Outcome (v3)
*   Peningkatan akurasi ke rentang **86% - 88%** (sesuai target awal).
*   Peningkatan F1-Score kelas *Hydrolase* menjadi **>0.70** (minimal), **>0.75** (ideal).
*   Kurva loss yang lebih stabil tanpa early stopping terlalu dini.
