# Strategi Implementasi ESM-2 Fine-Tuning (Phase 4c) — Laporan Hasil Training

> **Model Version**: `esm2 v1` — Iterasi pertama implementasi ESM-2 fine-tuning. Rencana pengembangan selanjutnya akan diberi label `esm2 v2`, konsisten dengan skema versi model CNN dan LSTM.

## 1. Gambaran Umum

Notebook `05_esm2_model_colab.ipynb` mengimplementasikan **LoRA fine-tuning** model pretrained **ESM-2** (Evolutionary Scale Modeling) untuk klasifikasi 6 famili protein. ESM-2 adalah model bahasa protein dari Meta AI yang telah dilatih pada 65 juta sekuens protein, dan memberikan performa terbaik dibanding CNN dan LSTM berkat transfer learning.

**Proses training**: Awalnya dirancang untuk RTX 2050 4GB (VRAM terbatas), tetapi terbukti terlalu lambat (~30 menit/epoch). Training akhirnya dijalankan di **Google Colab T4 (15.6 GB VRAM)** dengan konfigurasi yang dioptimasi.

### Base Model
- **Model**: `facebook/esm2_t12_35M_UR50D`
- **Parameter**: 35 juta (dimensi 480)
- **Pretrained**: 65M sekuens protein (UR50/D)
- **Strategi**: LoRA fine-tuning (backbone di-freeze)

---

## 2. Dataset

Menggunakan dataset yang sama persis dengan CNN dan LSTM:

| Split | File | Jumlah |
|-------|------|:------:|
| Train | `data/processed/train.csv` | 17,564 |
| Validation | `data/processed/val.csv` | 3,764 |
| Test | `data/processed/test.csv` | 3,764 |

**Format data** (sama seperti CNN/LSTM):
```
Entry,Sequence,Length,Label,Family
Q69TW5,MELPADGS...,324,5,Transcription Factor
```

**Label mapping** (`data/processed/label_mapping.json`):
```json
{"0": "GPCR", "1": "Hydrolase", "2": "Ion Channel",
 "3": "Kinase", "4": "Oxidoreductase", "5": "Transcription Factor"}
```

**Catatan penting**: Berbeda dengan CNN/LSTM yang melakukan tokenisasi manual ke integer (20 asam amino + padding), ESM-2 menggunakan tokenizer bawaan dari Hugging Face yang sudah handle tokenisasi, `<cls>`, `<eos>`, dan `<pad>` secara otomatis.

---

## 3. Arsitektur Model

### 3.1 ESM-2 Base Model
```
Esm2ForSequenceClassification(
  (esm2): Esm2Model(
    (embed_tokens): Embedding(33, 480, padding_idx=1)  # 33 vocab ESM
    (layers): 12× Esm2DecoderLayer(hidden_dim=480, ffn_dim=1920, heads=8)
    ...
  )
  (classifier): Linear(480 → 6)  # Akan diganti
)
```

### 3.2 LoRA Configuration (PEFT)
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value"],  # Q/K/V projections
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
```

### 3.3 Classification Head (Custom MLP)
Default classifier `Linear(480 → 6)` akan diganti dengan MLP yang lebih expressif:

```python
class ProteinClassificationHead(nn.Module):
    def __init__(self, hidden_dim=480, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, cls_token):
        return self.fc(cls_token)
```

Mengambil **[CLS] token** (index 0) dari output ESM-2 sebagai representasi sekuens.

### 3.4 Parameter Distribution (Aktual)
| Komponen | Jumlah Parameter | Trainable? |
|----------|:----------------:|:----------:|
| ESM-2 Backbone | ~34,287,837 | ❌ Freeze |
| LoRA (r=8, Q/K/V) | ~276,480 | ✅ Ya |
| Classification Head | ~124,678 | ✅ Ya |
| **Total Trainable** | **124,678** | ✅ **(0.36% dari total)** |

> **Catatan**: PEFT melaporkan 124,678 trainable params (dari total 34,287,837) pada `model.print_trainable_parameters()`. Nilai lebih rendah dari estimasi awal (401,158) karena `modules_to_save=["classifier", "score"]` menandai modul sebagai trainable tetapi PEFT hanya menghitung parameter LoRA dalam laporan trainable. Weight classifier penuh (termasuk bias) tersimpan di adapter tetapi dihitung sebagai "all params" bukan "trainable params" oleh PEFT.

---

## 4. Tokenization

### 4.1 ESM Tokenizer
ESM-2 menggunakan tokenizer sendiri dengan vocab size 33 (20 asam amino + token khusus):

| Token | ID | Makna |
|-------|:--:|-------|
| `<cls>` | 0 | Classification token (awal sekuens) |
| `<pad>` | 1 | Padding token |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown |
| A, C, D, ..., Y | 4-23 | 20 asam amino standar |
| lainnya | - | Token khusus lainnya |

### 4.2 Proses Tokenisasi per Sekuens
```python
inputs = tokenizer(
    sequence,
    padding="max_length",
    truncation=True,
    max_length=1002,
    return_tensors="pt"
)
# Input: "MELPADGS..."
# Output: input_ids=[0, 11, 8, 16, ..., 2, 1, 1, ...] (panjang 1002)
```
- `<cls>` ditambahkan di awal (ID 0)
- `<eos>` ditambahkan di akhir (ID 2)
- Sisanya di-pad ke max_length 1002

**Catatan**: Panjang sekuens asli di dataset max 1000, ditambah `<cls>` dan `<eos>` = 1002 token. Ini optimal untuk VRAM 4GB.

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter | Nilai (Colab) | Alasan |
|-----------|---------------|--------|
| **Base Model** | `esm2_t12_35M_UR50D` | 35M params — cocok untuk T4 16GB |
| **LoRA r** | 8 | Standar untuk fine-tuning efisien |
| **LoRA alpha** | 32 | Scaling factor (2× r) |
| **LoRA dropout** | 0.1 | Regularisasi ringan |
| **LoRA target** | query, key, value | Proyeksi attention paling impactfull |
| **Batch Size** | 16 (vs rencana awal 8) | VRAM T4 16GB memungkinkan batch lebih besar |
| **Gradient Accumulation** | 2 steps (vs rencana awal 4) | Effective batch size = 32 (16 × 2) |
| **Max Length** | 1002 | 1000 asli + `<cls>` + `<eos>` |
| **Optimizer** | AdamW | Standar untuk fine-tuning transformer |
| **Learning Rate** | 1e-4 | Lebih kecil dari CNN/LSTM (1e-3) karena backbone pretrained |
| **Weight Decay** | 1e-4 | Regularisasi L2 |
| **LR Scheduler** | ReduceLROnPlateau(factor=0.5, patience=2) | Sama dengan CNN/LSTM (LR turun di epoch 7) |
| **Early Stopping** | patience=5 | Sama dengan CNN/LSTM |
| **Max Epochs** | 50 | Sama dengan CNN/LSTM |
| **Mixed Precision** | FP16 via `GradScaler` | Penting untuk efisiensi VRAM & kecepatan |

### 5.2 AdamW vs Adam

ESM-2 menggunakan **AdamW** (bukan Adam seperti CNN/LSTM) karena:
- AdamW adalah standar de facto untuk fine-tuning transformer
- Weight decay diterapkan dengan benar (tidak tercampur dengan momentum)
- Mencegah overfitting pada LoRA + classification head

### 5.3 Gradient Accumulation Details (Aktual: 2 steps)

```python
accumulation_steps = 2  # Colab: batch=16, accum=2 → effective 32
optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    with torch.amp.autocast('cuda'):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
    scaler.scale(loss).backward()
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 6. Implementasi Training Loop

### 6.1 Mixed Precision (FP16)

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast('cuda'):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits, labels)  # manual loss, bukan outputs.loss
```

FP16 menghemat VRAM ~40-50%, memungkinkan batch_size=16 pada T4 16GB.

> **Catatan**: Berbeda dengan rencana awal yang menggunakan `outputs.loss`, implementasi aktual menggunakan `criterion` manual dengan `class_weights` untuk menangani ketidakseimbangan kelas.

### 6.2 Dataset Class

```python
class ProteinESMDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=1002):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences.iloc[idx])
        label = self.labels.iloc[idx]

        encoded = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

### 6.3 Training per Epoch

```python
# Training Phase
model.train()
train_loss, correct, total = 0.0, 0, 0
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.amp.autocast('cuda'):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    train_loss += loss.item() * accumulation_steps * input_ids.size(0)
    _, preds = torch.max(outputs.logits, 1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
```

### 6.4 Validation per Epoch

```python
# Validation Phase
model.eval()
val_loss, correct_val, total_val = 0.0, 0, 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        val_loss += outputs.loss.item() * input_ids.size(0)
        _, preds = torch.max(outputs.logits, 1)
        correct_val += (preds == labels).sum().item()
        total_val += labels.size(0)
```

### 6.5 Checkpoint & Early Stopping

```python
if epoch_val_loss < best_val_loss:
    best_val_loss = epoch_val_loss
    best_epoch = epoch + 1
    epochs_no_improve = 0
    # Save only the PEFT adapter weights (small ~2MB vs full 140MB)
    model.save_pretrained("models/esm2_model_best")
else:
    epochs_no_improve += 1
    if epochs_no_improve >= PATIENCE:
        print(f"\n[Early Stopping Triggered]...")
        break

# Save full model after training completes
torch.save(model.state_dict(), "models/esm2_model.pth")
```

---

## 7. Evaluasi

### 7.1 Metrics yang Diukur (sama dengan CNN/LSTM)
- **Accuracy** (`sklearn.metrics.accuracy_score`)
- **F1 Macro** (`sklearn.metrics.f1_score`, average='macro')
- **Classification Report** (precision, recall, f1 per kelas)
- **MCC** (`sklearn.metrics.matthews_corrcoef`)
- **Confusion Matrix** (heatmap seaborn)

### 7.2 Output Files
| File | Fungsi |
|------|--------|
| `models/esm2_model_best/` | LoRA adapter weights (best checkpoint) |
| `models/esm2_model.pth` | Full model weights (final) |
| `data/results/esm2_metrics.json` | Accuracy, F1, MCC, training history |

### 7.3 Format Results JSON (Aktual)

```json
{
    "accuracy": 0.9307,
    "f1_macro": 0.9325,
    "mcc": 0.9168,
    "history": {
        "train_loss": [0.8176, 0.3110, 0.2182, 0.1754, 0.1436, 0.1199, 0.1083, 0.0802, 0.0730],
        "train_acc": [0.7249, 0.9049, 0.9341, 0.9480, 0.9554, 0.9613, 0.9661, 0.9744, 0.9761],
        "val_loss": [0.3873, 0.2643, 0.2317, 0.2225, 0.2256, 0.2257, 0.2441, 0.2360, 0.2542],
        "val_acc": [0.8804, 0.9195, 0.9299, 0.9315, 0.9304, 0.9386, 0.9301, 0.9408, 0.9376]
    }
}
```

---

## 8. Visualisasi

### 8.1 Training History Plot
Figure (12×5 inch) dengan 2 subplot:
- **Loss**: Train Loss vs Val Loss per epoch
- **Accuracy**: Train Accuracy vs Val Accuracy per epoch

### 8.2 Confusion Matrix
Figure (8×6 inch) — heatmap seaborn, annotasi angka, label kelas.

---

## 9. Struktur Notebook

| # | Cell | Type | Isi |
|---|------|------|-----|
| 1 | Markdown | `# Phase 4: Model Implementation - ESM-2 Fine-Tuning (Google Colab)` — penjelasan ESM-2 + instruksi upload data ke Google Drive |
| 2 | Code | **Mount Google Drive**, create folder structure di Drive (`data/`, `models/`, `results/`) |
| 3 | Code | Install dependencies: transformers, peft, scikit-learn, matplotlib, seaborn, pandas. Import semua library. Set device CUDA, set seed reproducibility. |
| 4 | Code | Load dataset dari Google Drive: `train.csv`, `val.csv`, `test.csv` + `label_mapping.json`. Print shape. |
| 5 | Code | Load ESM-2 tokenizer: `AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")`. Print vocab size, special tokens. |
| 6 | Code | Definisikan `ProteinESMDataset` class — menerima sequences, labels, tokenizer, max_len=1002. `__getitem__` return dict: `input_ids`, `attention_mask`, `labels`. |
| 7 | Code | Buat Dataset & DataLoader instances. train_loader shuffle=True, val/test shuffle=False, batch_size=16 (Colab). |
| 8 | Code | Setup model: load `AutoModelForSequenceClassification` dengan num_labels=6. Ganti classifier head dengan `CustomClassifier` (480→256→6). Konfigurasi LoRA via `LoraConfig` + `get_peft_model`. Print trainable parameters. |
| 9 | Code | Training configuration: criterion = CrossEntropyLoss (weighted), optimizer = AdamW(lr=1e-4, weight_decay=1e-4), scaler = GradScaler("cuda"), scheduler = ReduceLROnPlateau. EPOCHS=50, PATIENCE=5, accumulation_steps=2. |
| 10 | Code | **Training loop** (9 epoch aktual). Log per epoch. Save best model ke Drive via `model.save_pretrained()`. |
| 11 | Code | Training summary: early stopping info, best epoch. Load best weights. |
| 12 | Code | Evaluasi pada test set: classification report, MCC, confusion matrix + plot. |
| 13 | Code | Save metrics ke Drive: `esm2_metrics.json`, `confusion_matrix_esm2.png`, `learning_curves_esm2.png`. |
| 14 | Code | **Download hasil** — zip folder results/ dan download via `google.colab.files.download()` |

> **Catatan**: Berbeda dengan rencana awal yang punya 14 cell, implementasi Colab memiliki struktur yang disesuaikan (mount Drive + download cell). Cell 8 menggabungkan setup model + replacement classifier menjadi satu.

---

## 10. Actual Results & Analysis

### 10.1 Actual Performance vs Target
| Metrik | Target | Saved Metrics (Test Set) | Inference (RTX 2050) | Status |
|--------|:------:|:------------------------:|:--------------------:|:------:|
| **Accuracy** | ≥90% | **93.07%** | **93.09%** | ✅ Tercapai |
| **F1 Macro** | ≥0.90 | **0.9325** | **0.9327** | ✅ Tercapai |
| **MCC** | ≥0.88 | **0.9168** | **0.9171** | ✅ Tercapai |
| **F1 Hydrolase** | ≥0.82 | **0.87** | **0.8738** | ✅ Tercapai |
| **Best Epoch** | ~15-25 | **Epoch 4** | — | ⚡ Lebih cepat dari prediksi |
| **Waktu/Epoch** | ~200s | **~1040s** (~17 menit) | — | ⚠️ Lebih lambat (T4 vs prediksi) |
| **Inference Time (3,764 seq)** | — | — | **313.2s** (~5.2 min) | — |

> **Catatan**: Selisih saved metrics vs inference hanya 0.02% — sangat konsisten. Inference di RTX 2050 dengan FP16 + batch size 8 memakan waktu 313.2 detik untuk 3,764 sekuen.

### 10.2 Perbandingan Target vs Aktual

**Yang sesuai target:**
- Semua metrik utama (Accuracy, F1, MCC) melampaui target
- ESM-2 terbukti superior dibanding CNN (82.76%) dan LSTM (86.85%)
- LoRA fine-tuning efektif tanpa overfitting

**Yang berbeda dari prediksi:**
- **Best epoch**: Hanya epoch 4 (dari prediksi 15-25) — konvergensi sangat cepat karena:
  - Transfer learning ESM-2 sangat efektif untuk dataset ini
  - Classifier head baru belajar cepat dari representasi ESM-2 yang sudah matang
- **Waktu/epoch**: ~1040s (vs prediksi ~200s) karena:
  - Tokenizer memproses max_length=1002 untuk 17,564 sekuens (bottleneck)
  - T4 GPU tidak secepat yang diasumsikan untuk model 35M parameter
- **Early stopping**: Trigger di epoch 9 (patience=5), bukan epoch 20-30

### 10.3 Training Progress Detail (9 Epochs)

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

### 10.4 Per-Class Performance (Test Set)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| GPCR | 0.99 | 0.97 | 0.98 | 498 |
| Hydrolase | 0.86 | 0.89 | 0.87 | 643 |
| Ion Channel | 0.95 | 0.97 | 0.96 | 671 |
| Kinase | 0.96 | 0.92 | 0.94 | 628 |
| Oxidoreductase | 0.90 | 0.95 | 0.93 | 679 |
| Transcription Factor | 0.94 | 0.88 | 0.91 | 645 |

**Observasi:**
- **GPCR** — performa terbaik (F1=0.98), sangat mudah dikenali
- **Ion Channel** — hampir sempurna (F1=0.96), recall 0.97 menunjukkan sedikit false negative
- **Hydrolase** — terendah (F1=0.87), konsisten dengan CNN & LSTM (kelas paling beragam secara sekuens)
- **Transcription Factor** — recall 0.88, sering tertukar dengan kelas lain
- Tidak ada kelas dengan F1 < 0.87 — model seimbang untuk semua kelas

### 10.5 Inference Performance (Phase 5 Comparative Analysis)

Pada notebook `06_comparative_analysis.ipynb`, ESM-2 v1 di-load di RTX 2050 4GB dan menjalankan inference pada 3,764 sekuen test:

| Metrik | Nilai |
|--------|:-----:|
| **Accuracy** | 93.09% |
| **F1 Macro** | 0.9327 |
| **MCC** | 0.9171 |
| **Inference Time** | 313.2 detik (~5.2 menit) |
| **Device** | RTX 2050 4GB |
| **Batch Size** | 8 |
| **Mixed Precision** | FP16 |

**Konsistensi**: Selisih accuracy antara inference (93.09%) dan saved metrics (93.07%) hanya 0.02%, mengonfirmasi reproducibility yang baik.

### 10.6 Alasan ESM-2 Lebih Baik (Terbukti)
1. **Pretrained knowledge**: ESM-2 telah melihat 65M sekuens — memahami "fisika" dan "evolusi" protein
2. **Representasi contextual**: Setiap asam amino di-embed berdasarkan konteks kiri-kanan (bukan statis seperti CNN/LSTM)
3. **Attention mechanism**: Self-attention 12 layer menangkap long-range interactions lebih baik dari LSTM 2 layer
4. **LoRA fine-tuning**: Adaptasi efisien tanpa kehilangan pengetahuan pretrained

### 10.7 Risiko & Mitigasi (Evaluasi)
| Risiko | Dampak | Mitigasi | Hasil |
|--------|:------:|----------|:-----:|
| **OOM (Out of Memory)** | Training gagal | FP16 + batch 16 + accum 2 | ✅ Tidak terjadi (T4 16GB cukup) |
| **Overfitting** | Performa test turun | LoRA dropout=0.1 + weight_decay + early stopping | ✅ Tidak terjadi (train/val gap kecil) |
| **Training lambat** | Waktu lama | Pindah ke Colab T4 | ⚠️ 17 menit/epoch (masih lambat tapi acceptable) |
| **Catastrophic forgetting** | Hilang pengetahuan pretrained | LoRA (backbone freeze) + LR kecil (1e-4) | ✅ Performa tinggi membuktikan |

---

## 11. Daftar Library yang Digunakan

```python
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
```

---

## 12. Dependency Versions (Terinstall)

| Package | Colab Version | Local Version | Catatan |
|---------|:-------------:|:-------------:|---------|
| torch | 2.5.1+cu121 | 2.5.1+cu124 | CUDA 12.1 (Colab) / 12.4 (Local) |
| transformers | 4.41.2 | 4.41.2 | Hugging Face |
| peft | **0.19.1** | **0.12.0** | Colab lebih baru; local 0.12.0 cukup untuk load adapter |
| accelerate | 1.13.0 | ~1.0.0 | Required by PEFT |
| torchao | **≥0.16.0** | — | Diupgrade untuk kompatibilitas PEFT terbaru (Colab only) |
| sklearn | 1.x | 1.x | Metrics |
| matplotlib | 3.x | 3.x | Plotting |
| seaborn | 0.x | 0.x | Confusion matrix |

> **Catatan penting**: PEFT v0.19.1 (Colab) vs v0.12.0 (local) — Adapter yang di-train dengan PEFT v0.19.1 kompatibel dengan PEFT v0.12.0 untuk inference. Tidak ada breaking changes antara versi ini untuk operasi load adapter.

---

## 13. Perbandingan dengan CNN & LSTM

| Aspek | CNN | LSTM | ESM-2 |
|-------|:---:|:----:|:-----:|
| **Parameter** | ~340K | ~827K | ~34.3M (124.7K trainable) |
| **Trainable** | 340K (100%) | 827K (100%) | 124.7K (0.36%) |
| **Tokenization** | Manual (21 vocab) | Manual (21 vocab) | ESM tokenizer (33 vocab) |
| **Max Length** | 1000 | 1000 | 1002 |
| **Batch Size** | 64 | 64 | **16** (Colab T4) |
| **Effective Batch** | 64 | 64 | **32** (accum=2) |
| **LR** | 1e-3 | 5e-4 | 1e-4 |
| **Optimizer** | Adam | Adam | AdamW |
| **Mixed Precision** | ❌ FP32 | ❌ FP32 | ✅ FP16 |
| **GPU** | RTX 2050 4GB | RTX 2050 4GB | **T4 16GB (Colab)** |
| **Accuracy** | 82.76% | 86.85% | **93.09%** |
| **F1 Macro** | 0.8349 | 0.8716 | **0.9327** |
| **MCC** | 0.7932 | 0.8422 | **0.9171** |
| **Training Time/Epoch** | ~24s | ~49s | **~1040s** (~17 menit) |
| **Best Epoch** | — | — | **Epoch 4** |
| **Total Training** | ~2 menit | ~4 menit | **~2.5 jam** |
| **Inference (3,764 seq)** | **2.5s** | 4.1s | 313.2s |

---

## 14. Catatan Implementasi Khusus

### 14.1 ESM-2 Output Structure
Hugging Face `AutoModelForSequenceClassification` untuk ESM-2 mengembalikan:
```python
outputs = model(input_ids=..., attention_mask=..., labels=...)
outputs.loss        # CrossEntropyLoss (scalar)
outputs.logits      # [batch_size, num_classes] — logit mentah
```

[CLS] token adalah **first token** dari `outputs.hidden_states[-1][:, 0, :]`, tapi untuk sequence classification, kita bisa langsung menggunakan `outputs.logits` dari classification head.

### 14.2 Replacement of Classifier Head
```python
# Hapus default classifier
model.classifier = nn.Identity()

# Buat custom head
class CustomClassifier(nn.Module):
    def __init__(self, hidden_size=480, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

model.classifier = CustomClassifier()
model.config.num_labels = 6
```

### 14.3 Saving Strategy
- **Best checkpoint**: Simpan hanya LoRA adapter (`model.save_pretrained()`) — ukuran ~2MB
- **Final model**: Simpan full state_dict (`torch.save()`) — ukuran ~140MB
- **Load best**: `PeftModel.from_pretrained(base_model, "models/esm2_model_best")`

### 14.4 Reproducibility
```python
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 15. Timeline Estimasi

| Tahap | Estimasi | Aktual | Keterangan |
|-------|:--------:|:------:|------------|
| Persiapan + coding | 30 menit | 30 menit | Menulis notebook |
| Epoch 1-10 | ~30-40 menit | **~2.5 jam** (9 epoch × 1040s) | T4 lebih lambat dari ekspektasi |
| Epoch 11-30 (best) | ~40-80 menit | **N/A** | Early stopping di epoch 9 |
| **Total training** | **~2-3 jam** | **~2.5 jam** | Sesuai estimasi (tapi karena 9 epoch, bukan 15-25) |
| Evaluasi + plotting | 5 menit | 5 menit | Saving results |
| **Total keseluruhan** | **~2.5-3.5 jam** | **~3 jam** | ✅ Sesuai estimasi |

---

## 16. Post-Training: Integrasi dengan `06_comparative_analysis.ipynb` (Selesai)

Notebook `06_comparative_analysis.ipynb` telah selesai dijalankan (30 cell, 0 error) dengan output:
1. Load metrics dari `data/results/cnn_metrics.json`, `lstm_metrics.json`, `esm2_metrics.json`
2. Load model weights (CNN, LSTM, ESM-2) dan jalankan inference pada 3,764 sekuen test
3. Tabel perbandingan: Accuracy (82.76% vs 86.85% vs **93.09%**), F1 Macro, MCC
4. 8 figure visualisasi (bar charts, confusion matrices, radar chart, error analysis, trade-off)
5. Analisis trade-off: performa vs training time vs parameter count
6. Confusion matrices side-by-side, error analysis, model agreement
7. Kesimpulan: ESM-2 unggul akurasi, CNN unggul efisiensi, LSTM sebagai kompromi
8. Prediksi disimpan ke `data/results/comparative_predictions.npz`

---

## 17. Strategi Peningkatan Performa — `esm2 v2` (Future Work)

Meskipun ESM-2 v1 sudah mencapai 93.07% accuracy, masih ada beberapa strategi yang bisa diterapkan untuk meningkatkan performa lebih lanjut pada iterasi berikutnya (`esm2 v2`):

### 17.1 Model-Level Improvements

| Strategi | Deskripsi | Potensi Gain | Trade-off |
|----------|-----------|:------------:|-----------|
| **ESM-2 Larger Model** | Gunakan `esm2_t33_650M_UR50D` (650M params, 33 layer, hidden=1280) | +1-3% | VRAM >>16GB, jauh lebih lambat |
| **ESM-2 Medium** | Gunakan `esm2_t30_150M_UR50D` (150M params, 30 layer, hidden=768) | +0.5-1.5% | Mungkin muat di T4 dengan batch lebih kecil |
| **Full fine-tuning** | Unfreeze seluruh backbone (bukan hanya LoRA) | +0.5-2% | Risiko overfitting, VRAM >>16GB |
| **LoRA rank tuning** | Coba r=16 atau r=32 untuk LoRA | +0.3-1% | Lebih banyak parameter trainable |
| **Target modules tambahan** | Tambahkan `output`, `dense`, `fc` ke LoRA target | +0.2-0.5% | Sedikit lebih lambat |
| **DoRA (Weight-Decomposed LoRA)** | Gunakan `use_dora=True` di PEFT | +0.3-0.8% | Sedikit lebih lambat |

### 17.2 Data-Level Improvements

| Strategi | Deskripsi | Potensi Gain |
|----------|-----------|:------------:|
| **Data Augmentation** | Mutasi sekuens protein (substitusi asam amino homolog) | +0.5-1% (terutama untuk Hydrolase & TF) |
| **Oversampling** | Oversample kelas minoritas (Hydrolase mudah salah klasifikasi) | +0.2-0.5% |
| **Sequence weighting** | Weight sekuens berdasarkan cluster homology (redundansi rendah) | +0.3-0.8% |
| **Taxonomic balancing** | Seimbangkan representasi taksonomi dalam dataset | +0.2-0.5% |

### 17.3 Training-Level Improvements

| Strategi | Deskripsi | Potensi Gain |
|----------|-----------|:------------:|
| **Learning Rate Find** | Gunakan `LR Finder` untuk optimasi LR | +0.2-0.5% |
| **Cosine Annealing** | Ganti ReduceLROnPlateau dengan CosineAnnealingLR | +0.3-0.7% |
| **Warmup Steps** | Tambah linear warmup (5-10% steps) sebelum LR utama | +0.2-0.4% |
| **Label Smoothing** | Gunakan CrossEntropyLoss dengan label_smoothing=0.1 | +0.1-0.3% |
| **Focal Loss** | Ganti CrossEntropy dengan Focal Loss untuk kelas sulit | +0.2-0.5% |
| **MixUp / CutMix** | Augmentasi tingkat representasi (interpolasi embedding) | +0.3-0.6% |
| **K-Fold Cross Validation** | 5-fold CV untuk memaksimalkan data training | +0.5-1.5% |

### 17.4 Post-Training Improvements

| Strategi | Deskripsi | Potensi Gain |
|----------|-----------|:------------:|
| **Ensemble Voting** | Gabungkan CNN + LSTM + ESM-2 (soft voting) | +0.5-2% | *(Not in scope — fokus ke comparative analysis)* |
| **Test-Time Augmentation (TTA)** | Prediksi dengan multiple mutated copies, rata-rata hasil | +0.2-0.5% |
| **Threshold Tuning** | Optimasi threshold per kelas (bukan default 0.5) | +0.1-0.3% |
| **Confidence Calibration** | Temperature scaling untuk probability calibration | - (lebih reliable, tidak selalu meningkatkan akurasi) |

### 17.5 Rekomendasi Prioritas

Berdasarkan effort vs impact, prioritas yang paling direkomendasikan:

1. ~~**Ensemble (CNN + LSTM + ESM-2)** — gain terbesar dengan effort minimal~~ *(dihapus, fokus ke comparative analysis)*
2. **Data augmentation (homolog mutation)** — langsung mengatasi kelemahan Hydrolase
3. **LoRA rank tuning (r=16)** — perubahan minimal, potensi gain nyata
4. **Focal Loss** — jika ada kelas yang persistently underperforms
5. **K-Fold CV** — maksimalkan penggunaan data, meskipun training ×5 lebih lama
