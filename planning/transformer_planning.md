# Transformer From Scratch — Strategi Pengembangan

## Latar Belakang

Saat ini komparasi memiliki gap: **CNN** (konvolusi/pola lokal), **LSTM** (recurrence/sekuensial), **ESM-2** (transfer learning/pretrained). Transformer from-scratch mengisolasi **self-attention murni tanpa pretraining** — menjawab apakah arsitektur transformer saja cukup efektif untuk klasifikasi protein, atau memang butuh pretraining skala besar seperti ESM-2.

## Perbandingan Parameter & Kompleksitas

| Model | Total Params | Waktu/Epoch | VRAM | Akurasi |
|-------|:-----------:|:-----------:|:----:|:-------:|
| CNN | ~340K | ~24s | Rendah | 82.76% |
| LSTM | ~827K | ~49s | Rendah | 86.85% |
| **Transformer (est.)** | **~2–5M** | **?** | **?** | **?** |
| ESM-2 | 34.3M (124.7K trainable) | ~1040s (T4) | 15.6GB | 93.09% |

## Arsitektur v1 (Proposal)

### Komponen

| Komponen | Pilihan | Alasan |
|----------|---------|--------|
| Token Embedding | `nn.Embedding(21, 256), padding_idx=0` | Sama dengan CNN/LSTM (20 asam amino + PAD) |
| Max Sequence Length | 1000 | Konsisten dengan preprocessing yang sudah ada |
| Positional Encoding | **Sinusoidal** | Tanpa parameter, stabil untuk from-scratch, generalisasi ke panjang seq berbeda |
| Encoder Layers | 6 | Cukup dalam untuk menangkap hierarki motif protein |
| Attention Heads | 8 | d_model (256) / 8 = 32 dim per head — standar |
| FFN Hidden | 1024 | 4× d_model — standard transformer |
| Dropout | 0.2 (attention), 0.3 (FFN) | Regularisasi di rentang aman |
| Pooling | **Max + Average Combined** | Konsisten dengan CNN & LSTM (hasil concat = 512 dim) |
| Classifier Head | `Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 6)` | Sama dengan CNN/LSTM |

### Estimasi Parameter

| Komponen | Perhitungan | Parameter |
|----------|-------------|:---------:|
| Token Embedding | 21 × 256 | 5,376 |
| Positional Encoding | 0 (fixed) | 0 |
| Per-Layer Attention | QKV: 3×(256×256) + Output: 256×256 + LN: 2×256 | 200,704 |
| Per-Layer FFN | 256×1024 + 1024×256 + LN: 2×256 | 524,800 |
| Per-Layer Total | — | **725,504** |
| 6 Layers | 725,504 × 6 | 4,353,024 |
| Classifier Head | 512×256 + 256×6 | 132,608 |
| **Total** | | **~4.49M** |

Antara LSTM (~827K) dan ESM-2 (34.3M). Posisi yang baik untuk menguji trade-off parameter vs performa.

### Estimasi VRAM (RTX 2050 4GB)

Komponen terbesar: **attention score matrix** — `batch × heads × seq² × layers × precision`

| Batch Size | Attention Memory (FP32) | Total Estimasi | VRAM 4GB? |
|:----------:|:-----------------------:|:--------------:|:---------:|
| 8 | 8 × 8 × 1e6 × 6 × 4 = 1.5GB | ~2.0GB | ✅ Aman |
| 16 | 16 × 8 × 1e6 × 6 × 4 = 3.0GB | ~4.0GB | ⚠️ Limit |
| 32 | 6.0GB | ~8.0GB | ❌ Tidak |

**Rekomendasi:** Batch size 8, gradient accumulation 4 (effective batch 32). Kalau memungkinkan, coba FP16 mixed precision.

## Iterasi

### v1 — Baseline Transformer

**Target Arsitektur:**
- 6 layer TransformerEncoder (`nn.TransformerEncoderLayer`)
- d_model=256, nhead=8, dim_feedforward=1024
- Sinusoidal positional encoding
- Max + Average pooling
- AdamW (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau
- Batch size 8, gradient accumulation 4
- Early stopping patience=5, max 50 epochs
- seed=42 untuk reproducibility

**Target Metrik:**
- Akurasi: >85% (minimal di atas LSTM? atau realistis?)
- F1 Macro: >0.85

**Risiko:**
- Overfitting — dataset 17,564 training seq, model 4.5M parameter (ratio ~4:1). Butuh regularisasi lebih agresif dari CNN/LSTM.
- Training lambat — attention O(n²) untuk seq 1000
- VRAM limit — perlu FP16 atau batch kecil

### Rencana Setelah v1 (Prioritas)

1. **Hyperparameter tuning**: d_model (128/256/512), layers (4/6/8), dropout tuning
2. **RoPE** jika v1 dengan Sinusoidal kurang memuaskan
3. **Focal Loss / Label Smoothing** — jika ada kelas underperforming
4. **Pre-LayerNorm** (vs Post-LayerNorm default) — biasanya lebih stabil untuk from-scratch training
5. **Gradient Clipping** — penting untuk transformer from-scratch (sama seperti LSTM)

## Perbandingan yang Diharapkan

Setelah model selesai, empat model bisa dibandingkan dalam spektrum:

```
Kompleksitas Rendah ←─────────────────────────────→ Kompleksitas Tinggi
      CNN           LSTM     Transformer (scratch)      ESM-2
   ~340K params    ~827K         ~4.5M              34.3M (124.7K trainable)
   ~24s/epoch      ~49s           ?                 ~1040s (T4)
   Pola lokal      Sekuensial    Self-attention     Transfer learning
```

Ini mengisi celah yang ada: arsitektur transformer murni tanpa pretraining di antara LSTM dan ESM-2 dalam spektrum parameter dan kompleksitas.
