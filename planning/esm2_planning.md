# ESM-2 — Strategi Fine-Tuning

## Iterasi

### Rencana Awal
- **Base Model:** `facebook/esm2_t12_35M_UR50D` (35M parameter, hidden=480)
- **Training:** Lokal RTX 2050 4GB
- **Strategi:** LoRA (r=8, alpha=32, dropout=0.1) pada Q/K/V projections
- **Batch Size:** 8 (maksimal untuk 4GB VRAM), accum=4 (effective 32)

### Realisasi (v1 — `05_esm2_model_colab.ipynb`)
RTX 2050 4GB terbukti terlalu lambat (~30 menit/epoch) → pindah ke **Google Colab T4 (15.6 GB VRAM)**.

**Perubahan dari rencana awal:**
| Parameter | Rencana | Realisasi | Alasan |
|-----------|:-------:|:---------:|--------|
| Batch Size | 8 | **16** | VRAM T4 lebih besar |
| Accum Steps | 4 | **2** | Effective batch tetap 32 |
| Max Length | 1000 | **1002** | 1000 seq + `<cls>` + `<eos>` |
| Loss Function | `outputs.loss` | **Manual CrossEntropy + class_weights** | Menangani imbalance kelas |
| Saving | Full only | **LoRA adapter** (best) + **Full** (final) | Ukuran adapter ~2MB vs full ~140MB |

### Arsitektur
- **LoRA:** r=8, alpha=32, dropout=0.1, target_modules=["query", "key", "value"]
- **Classifier Head:** `CustomClassifier` MLP 480→256→6 (ReLU + Dropout 0.3)
- **Trainable Parameters:** 124,678 (0.36% dari total 34,3M)
- **Mixed Precision:** FP16 via `torch.amp.GradScaler`
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau, early stopping patience=5

### Training Progress (9 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|:-----:|:----------:|:---------:|:--------:|:-------:|:--:|
| 1 | 0.8176 | 0.7249 | 0.3873 | 0.8804 | 1e-4 |
| 2 | 0.3110 | 0.9049 | 0.2643 | 0.9195 | 1e-4 |
| 3 | 0.2182 | 0.9341 | 0.2317 | 0.9299 | 1e-4 |
| **4** | **0.1754** | **0.9480** | **0.2225** | **0.9315** | 1e-4 |
| 5 | 0.1436 | 0.9554 | 0.2256 | 0.9304 | 1e-4 |
| 6 | 0.1199 | 0.9613 | 0.2257 | 0.9386 | 1e-4 |
| 7 | 0.1083 | 0.9661 | 0.2441 | 0.9301 | 5e-5 |
| 8 | 0.0802 | 0.9744 | 0.2360 | 0.9408 | 5e-5 |
| 9 | 0.0730 | 0.9761 | 0.2542 | 0.9376 | 5e-5 |

**Best Epoch:** 4 (early stopped at epoch 9). Konvergensi sangat cepat — transfer learning ESM-2 sangat efektif untuk dataset ini.

## Hasil

### Performa Final

| Metrik | Target | Saved Metrics (Test) | Inference (RTX 2050) | Status |
|--------|:------:|:--------------------:|:--------------------:|:------:|
| **Accuracy** | ≥90% | **93.07%** | **93.09%** | ✅ |
| **F1 Macro** | ≥0.90 | **0.9325** | **0.9327** | ✅ |
| **MCC** | ≥0.88 | **0.9168** | **0.9171** | ✅ |
| **F1 Hydrolase** | ≥0.82 | **0.87** | **0.8738** | ✅ |

### Per-Class Performance (Test Set)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| GPCR | 0.99 | 0.97 | 0.98 | 498 |
| Hydrolase | 0.86 | 0.89 | 0.87 | 643 |
| Ion Channel | 0.95 | 0.97 | 0.96 | 671 |
| Kinase | 0.96 | 0.92 | 0.94 | 628 |
| Oxidoreductase | 0.90 | 0.95 | 0.93 | 679 |
| Transcription Factor | 0.94 | 0.88 | 0.91 | 645 |

Semua kelas >0.87 F1. Tidak ada kelas tertinggal — berbeda dengan CNN (Hydrolase 0.61) dan LSTM (Hydrolase 0.75).

### Inference Performance (RTX 2050)
- **Waktu:** 313.2 detik (~5.2 menit) untuk 3,764 sekuen (batch=8, FP16)
- **Konsistensi:** Selisih accuracy saved vs inference hanya 0.02%

### Perbandingan dengan CNN & LSTM

| Aspek | CNN | LSTM | ESM-2 |
|-------|:---:|:----:|:-----:|
| Parameter | ~340K | ~827K | ~34.3M (124.7K trainable) |
| Tokenization | Manual (21 vocab) | Manual (21 vocab) | ESM tokenizer (33 vocab) |
| Mixed Precision | FP32 | FP32 | FP16 |
| Accuracy | 82.76% | 86.85% | **93.09%** |
| Training/Epoch | ~24s | ~49s | ~1040s (Colab T4) |
| Inference (3764 seq) | **2.5s** | 4.1s | 313.2s |

### Catatan Penting
- **PEFT cross-version compatibility:** Adapter di-train dengan PEFT v0.19.1 (Colab), kompatibel dengan PEFT v0.12.0 (local) untuk inference
- **Notebook lokal (`05_esm2_model_local.ipynb`)** ditinggalkan — terlalu lambat (KeyboardInterrupt di epoch 2)
- **Model loading pertama lambat** (~30s) karena download base model dari Hugging Face
- **`modules_to_save=["classifier", "score"]** — PEFT melaporkan 124,678 trainable params (hitung parameter LoRA saja, classifier penuh disimpan sebagai "all params")

## Rencana Selanjutnya (v2)

### Prioritas Tinggi
1. **Data augmentation** (homolog mutation) — langsung mengatasi kelemahan Hydrolase
2. **LoRA rank tuning** (r=16) — perubahan minimal, potensi gain nyata
3. **Focal Loss** — jika ada kelas yang persistently underperforms

### Prioritas Sedang
- ESM-2 medium (`esm2_t30_150M_UR50D`, 150M params) — mungkin muat di T4
- Cosine annealing LR + warmup steps
- K-Fold Cross Validation (5-fold)

### Prioritas Rendah
- DoRA (Weight-Decomposed LoRA)
- Test-Time Augmentation
- Threshold tuning per kelas
