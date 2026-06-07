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

### Catatan: Perubahan Data Split

Data split diubah dari **80:20** (20,073/5,019) menjadi **70:15:15** (17,564/3,764/3,764) untuk menyediakan validation set terpisah. Hasil di bawah menggunakan split baru dengan validation via `val_loader` dan evaluasi final pada test set unseen.

| Metrik | Target | v2 (80:20, old) | **v2 (70:15:15, baru)** | Delta |
|--------|--------|:----------------:|:------------------------:|:-----:|
| Akurasi | 86% - 88% | ~83.5% | **83.0%** | −0.5% |
| MCC | — | 0.8026 | **0.7932** | −0.009 |
| Macro Avg F1 | — | 0.84 | **0.83** | −0.01 |
| F1 Hydrolase | >0.75 | 0.65 | **0.61** | −0.04 |
| Best Epoch | — | 25 | **21** | −4 |
| Total Epoch | — | 30 | **26** | −4 |
| Waktu/Epoch | — | ~127-145s | **~24s** | −5× lebih cepat |

**Analisis v2 (70:15:15):**
- **Penurunan tipis akibat split change** — training set berkurang 12.5% (20,073→17,564).
- **Hydrolase paling terpukul** (F1 0.65→0.61). Precision sangat rendah (0.58), recall 0.65 — CNN kesulitan membedakan Hydrolase dari kelas lain.
- **GPCR dan Ion Channel stabil** (F1 >0.95) — kelas dengan pola sekuens khas tetap mudah dikenali.
- **Overfitting masih ada** — train acc akhir 83.5%, val acc 83.0% di best epoch 21. Gap kecil karena CNN memiliki parameter lebih sedikit dari LSTM.
- **Waktu training jauh lebih cepat** (~24s/epoch vs 145s sebelumnya), kemungkinan karena optimasi sistem atau beda kondisi runtime.

## 4. Rencana v3 — Strategi Lanjutan

### E. Data-Level Improvements
1. **Sequence Augmentation:**
   - Reverse complement setiap protein sequence (data 2x lipat)
   - Random mutation sampling (mengubah 1-2 asam amino secara acak, dengan probabilitas kecil)
2. **Upsampling Kelas Hydrolase:**
   - Oversample data Hydrolase di training set untuk memberi lebih banyak contoh
3. **Ubah Rasio Split:**
   - Split 75:12.5:12.5 (train 18,817) atau 80:10:10 (train 20,073) — mengembalikan data training yang hilang
   - Paling sederhana dan langsung mengatasi akar masalah (data training kurang)
4. **Filter Noise Sequence:**
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
   - Uji `OneCycleLR` — warmup lr kecil lalu cosine decay
3. **Weight Decay:**
   - Uji `weight_decay` = 1e-4 (saat ini), 5e-4, 1e-3

### I. Loss Function Experimentation
1. **Focal Loss:**
   - Mengganti CrossEntropyLoss dengan Focal Loss untuk fokus pada kelas sulit (Hydrolase)
   - Focal Loss secara otomatis mengurangi kontribusi loss dari kelas yang sudah mudah diklasifikasikan
   - Parameter awal: gamma=1.0 (sama dengan LSTM v5 yang berhasil)
2. **Label Smoothing:**
   - Menambahkan label smoothing ke CrossEntropyLoss untuk mencegah overconfidence

## 5. Prioritas Implementasi v3

| Prioritas | Langkah | Kategori | Effort | Impact | Recommended |
|:---------:|:--------|:--------:|:------:|:------:|:-----------:|
| 1 | **Ubah split (80:10:10 / 75:12.5:12.5)** | Data | ⚡ 1 jam | 🎯 Tinggi (langsung ke akar) | ✅ |
| 2 | **Focal Loss + Label Smoothing** | Loss | ⚡ 1 jam | 📈 Sedang | ✅ |
| 3 | **Hyperparameter Tuning** (dropout, LR, weight decay) | H-params | ⚡ 2-4 jam | 📈 Sedang | ✅ |
| 4 | **Sequence Augmentation** | Data | ⏳ 4-8 jam | 📈 Sedang | ⚠️ Hati-hati |
| 5 | **Multi-Layer Conv + Residual** | Arsitektur | ⏳ 4-8 jam | ❓ Tidak pasti | ❌ |
| 6 | **Attention Mechanism** | Arsitektur | ⏳ 4-8 jam | ❓ Tidak pasti | ❌ |
| 7 | **Pretrained Embeddings** | Representasi | ⏳ 8-16 jam | ❓ Tidak pasti | 🧪 Eksperimental |

## 6. Target Outcome (v3)
*   Peningkatan akurasi ke rentang **86% - 88%** (sesuai target awal).
*   Peningkatan F1-Score kelas *Hydrolase* menjadi **>0.70** (minimal), **>0.75** (ideal).
*   Kurva loss yang lebih stabil tanpa early stopping terlalu dini.
