# Rancangan Telegram Bot — Klasifikasi Fungsi Protein

## 1. Arsitektur Sistem

```
User ──→ Telegram Bot ──→ FastAPI Backend ──→ Model (.pth)
                        (python-telegram-bot)       │
                                                     ↓
                                              class_info.json
```

### Alur Prediksi

```
Input: "MKTAYIA..."
  ↓
Bot menerima pesan → kirim ke API POST /predict
  ↓
Backend preprocessing:
  1. Validasi: hanya 20 asam amino standar (ACDEFGHIKLMNPQRSTVWY)
  2. Filter panjang: max 1000 karakter
  3. Integer encoding (aa_to_int mapping)
  4. Padding ke panjang max (1000)
  ↓
Model inference (LSTM / sesuai pilihan)
  ↓
Response JSON:
  {
    "class": "Kinase",
    "label": 3,
    "confidence": 0.942,
    "probabilities": {...},
    "class_info": {
      "description": "...",
      "molecular_function": "...",
      "biological_process": "...",
      "cellular_location": "...",
      "organisms": "...",
      "function_in_organisms": "...",
      "benefits": "...",
      "examples": "..."
    }
  }
  ↓
Bot format pesan → kirim ke user
```

---

## 2. Struktur File

```
project-root/
├── api/
│   ├── main.py              # FastAPI app — endpoint /predict
│   ├── predict.py           # Logic preprocessing + inference
│   └── model_loader.py      # Load model .pth dari disk
├── telegram_bot.py           # Bot Telegram (python-telegram-bot v20+)
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── label_mapping.json
│   ├── raw/
│   │   └── protein_sequences.csv
│   ├── results/
│   │   ├── cnn_metrics.json
│   │   └── lstm_metrics.json
│   └── class_info.json       # [BARU] Info edukasi 6 famili protein
├── models/
│   ├── cnn_model.pth
│   ├── cnn_model_best.pth
│   ├── lstm_model.pth
│   └── lstm_model_best.pth
├── planning/
│   └── rancangan_telegram.md  # File ini
├── requirements.txt
└── .env                       # BOT_TOKEN, API_URL
```

---

## 3. Komponen Backend (FastAPI)

### 3.1 Endpoint

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/predict` | Prediksi class dari sequence |
| GET | `/health` | Health check |
| GET | `/models` | Info model yang tersedia |

### 3.2 Request `/predict`

```json
{
  "sequence": "MKTAYIA...",
  "model": "lstm"
}
```

Parameter `model`: `"lstm"` (default), `"cnn"`, `"esm2"`, atau `"all"` (semua model).

### 3.3 Response `/predict`

```json
{
  "success": true,
  "sequence_length": 45,
  "model_used": "lstm",
  "prediction": {
    "class": "Kinase",
    "label": 3,
    "confidence": 0.942,
    "probabilities": {
      "GPCR": 0.003,
      "Hydrolase": 0.021,
      "Ion Channel": 0.001,
      "Kinase": 0.942,
      "Oxidoreductase": 0.018,
      "Transcription Factor": 0.015
    }
  },
  "class_info": {
    "name": "Kinase",
    "description": "Kinase adalah enzim yang mengkatalisis transfer gugus fosfat dari ATP ke molekul target (fosforilasi). Fosforilasi adalah mekanisme kunci dalam mengatur aktivitas protein, lokasi, dan interaksinya dengan molekul lain.",
    "molecular_function": "ATP binding, protein kinase activity, transferase activity, nucleotide binding",
    "biological_process": "Protein phosphorylation, signal transduction, cell cycle regulation, apoptosis, cell differentiation, metabolism regulation",
    "cellular_location": "Cytoplasm, nucleus, cell membrane, mitochondrion",
    "organisms": "Homo sapiens (manusia), Mus musculus (tikus), Drosophila melanogaster (lalat buah), Arabidopsis thaliana (tumbuhan), Saccharomyces cerevisiae (ragi)",
    "function_in_organisms": "Pada manusia: mengatur pertumbuhan sel, sinyal insulin, dan divisi sel. Pada tumbuhan: merespons stres lingkungan dan mengatur pertumbuhan. Pada ragi: mengontrol siklus sel dan mating response.",
    "benefits": "Target utama obat kanker — lebih dari 70 inhibitor kinase telah disetujui FDA (imatinib, gefitinib, erlotinib, sorafenib). Juga diteliti untuk penyakit inflamasi dan neurodegeneratif.",
    "examples": "TTK_HUMAN, CLK3_HUMAN, DYRK2_HUMAN, MP2K3_HUMAN",
    "uniprot_example": "P33981"
  }
}
```

### 3.4 Validasi Input Backend

1. **Karakter**: hanya 20 asam amino standar (`A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`)
2. **Panjang**: 4 ≤ sequence ≤ 1000 (min mengikuti dataset, max sesuai preprocessing)
3. **Case**: uppercase (lowercase akan dikonversi otomatis)

---

## 4. Komponen Telegram Bot

### 4.1 Library

```bash
pip install python-telegram-bot==20.8
```

### 4.2 Command Handlers

| Command | Deskripsi |
|---------|-----------|
| `/start` | Sambutan + daftar 6 famili protein + contoh sequence |
| `/help` | Panduan lengkap cara menggunakan bot |
| `/about` | Info project, akurasi tiap model, technologies used |
| `/compare <sequence>` | Prediksi dari CNN + LSTM + ESM-2 sekaligus (side-by-side) |
| `/clear` | Hapus histori percakapan user |
| `/model <cnn/lstm/esm2>` | Ganti model default |

### 4.3 Message Handler

Pesan teks biasa (bukan command) yang berisi sequence protein → otomatis diprediksi.

#### Deteksi Sequence

```python
# Pola: huruf kapital A-Z saja, panjang 4-1000
import re
pattern = r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{4,1000}$'
if re.match(pattern, text):
    # ini sequence protein
    handle_prediction(text, update, context)
else:
    # bukan sequence, arahkan ke /help
    await update.message.reply_text("...")
```

### 4.4 Format Output Full

```
🧬 *Prediction Result*
Sequence: `MKTAYIA...` (45 aa)

🔮 *Predicted:* `Kinase` (94.2%)

📖 *Deskripsi*
Kinase adalah enzim yang mengkatalisis transfer gugus
fosfat dari ATP ke molekul target (fosforilasi).
Fosforilasi adalah mekanisme kunci dalam mengatur
aktivitas protein dan sinyal seluler.

⚙️ *Molecular Function*
• ATP binding
• Protein kinase activity
• Transferase activity

🔬 *Biological Process*
• Protein phosphorylation
• Signal transduction
• Cell cycle regulation
• Apoptosis

📍 *Cellular Location*
• Cytoplasm
• Nucleus
• Cell membrane

🧫 *Organisme Umum*
• Homo sapiens (manusia)
• Mus musculus (tikus)
• Drosophila melanogaster (lalat buah)
• Arabidopsis thaliana (tumbuhan)
• Saccharomyces cerevisiae (ragi)

🧪 *Fungsi pada Organisme*
• Manusia: mengatur pertumbuhan sel, sinyal insulin
• Tumbuhan: respons stres lingkungan
• Ragi: kontrol siklus sel

💊 *Manfaat & Relevansi*
Target utama obat kanker — 70+ inhibitor kinase
disetujui FDA (imatinib, gefitinib, erlotinib).

📋 *Contoh Protein*
TTK_HUMAN, CLK3_HUMAN, DYRK2_HUMAN

🔗 *Detail:* uniprot.org/uniprot/P33981

📊 *Confidence Breakdown:*
├─ GPCR              0.3%
├─ Hydrolase         2.1%
├─ Ion Channel       0.1%
├─ Kinase           94.2% ✅
├─ Oxidoreductase    1.8%
└─ Transcription F.  1.5%
```

### 4.5 Format `/start`

```
╔══════════════════════════════════╗
║   🧬 Protein Classifier Bot     ║
╚══════════════════════════════════╝

Selamat datang! Saya adalah bot untuk mengklasifikasikan
fungsi protein berdasarkan sekuens asam aminonya.

🔬 *Model Tersedia:*
• CNN (Conv1D) — 82.76% accuracy
• Bi-LSTM — 86.85% accuracy ✅ (terbaik)
• ESM-2 — (coming soon)

🧪 *6 Famili Protein:*
0️⃣ GPCR — G Protein-Coupled Receptor
1️⃣ Hydrolase — Enzim hidrolisis
2️⃣ Ion Channel — Saluran ion membran
3️⃣ Kinase — Enzim fosforilasi
4️⃣ Oxidoreductase — Enzim redoks
5️⃣ Transcription Factor — Regulasi transkripsi

📝 *Cara Pakai:*
Kirimkan sekuas protein (huruf kapital, A-Z):
Contoh: MKTAYIA...

📋 *Daftar Perintah:*
/start — Tampilkan pesan ini
/help — Panduan lengkap
/about — Info project & akurasi
/compare <seq> — Bandingkan semua model
/clear — Hapus histori chat
/model <cnn/lstm> — Ganti model default
```

### 4.6 Fitur `/clear`

Menggunakan `context.user_data.clear()` untuk menghapus data user yang tersimpan di session bot.

### 4.7 Conversation State (Opsional)

Jika ingin pengalaman lebih interaktif, bisa menggunakan `ConversationHandler`:

```
User: /start
Bot: Pilih famili yang ingin dipelajari:
     [GPCR] [Hydrolase] [Ion Channel] [Kinase] [Oxidoreductase] [Transcription Factor]
User: Kinase
Bot: (Menampilkan info lengkap Kinase)
     Ingin coba prediksi? Kirim sequence protein!
```

---

## 5. Data: `class_info.json`

### 5.1 Struktur

```json
{
  "0": {
    "name": "GPCR",
    "name_full": "G Protein-Coupled Receptor",
    "description": "GPCR adalah reseptor transmembran yang ...",
    "molecular_function": [
      "G protein-coupled receptor activity",
      "Signal transducer activity",
      ...
    ],
    "biological_process": [
      "Signal transduction",
      "Cell communication",
      ...
    ],
    "cellular_location": [
      "Cell membrane",
      "Integral component of membrane",
      ...
    ],
    "organisms": [
      {"name": "Homo sapiens", "common": "manusia", "note": ">800 GPCR dikenal"},
      ...
    ],
    "function_in_organisms": "GPCR pada manusia berperan dalam ...",
    "benefits": "Target ~35% obat yang disetujui FDA ...",
    "examples": ["OPSD_HUMAN", "ADRB2_HUMAN", ...],
    "uniprot_example": "P08100"
  },
  "1": { ... },
  ...
}
```

### 5.2 Konten per Famili (draft)

#### 0 — GPCR (G Protein-Coupled Receptor)

| Field | Konten |
|-------|--------|
| **Description** | Reseptor transmembran yang mendeteksi sinyal eksternal (cahaya, hormon, neurotransmitter) dan mengaktifkan jalur G protein di dalam sel. Struktur 7-transmembrane helix. |
| **Molecular Function** | G protein-coupled receptor activity, signal transducer activity, receptor binding |
| **Biological Process** | Signal transduction, cell-cell signaling, sensory perception, chemotaxis, hormone-mediated signaling |
| **Cellular Location** | Cell membrane, integral component of membrane, plasma membrane |
| **Organisms** | Manusia (>800 gen GPCR), tikus, lalat buah, C. elegans, zebrafish |
| **Function in Organisms** | Pada manusia: penglihatan (rhodopsin), penciuman, respons hormon, neurotransmisi. Pada lalat buah: chemosensation dan perilaku. |
| **Benefits** | ~35% obat yang disetujui FDA menargetkan GPCR — termasuk antihistamin, beta-blocker, opioid, dan antagonis angiotensin. |

#### 1 — Hydrolase

| Field | Konten |
|-------|--------|
| **Description** | Enzim yang mengkatalisis pemutusan ikatan kimia menggunakan molekul air (hidrolisis). Termasuk protease, esterase, lipase, fosfatase, dan glikosidase. |
| **Molecular Function** | Hydrolase activity, catalytic activity, peptidase activity, esterase activity |
| **Biological Process** | Metabolism, protein degradation, digestion, signal termination, detoxification |
| **Cellular Location** | Cytoplasm, lysosome, extracellular space, endoplasmic reticulum, membrane |
| **Organisms** | Universal — ditemukan di semua domain kehidupan (bakteri, arkea, eukariota) |
| **Function in Organisms** | Pada semua organisme: pencernaan nutrisi, daur ulang protein, aktivasi/terminasi sinyal. Pada bakteri: virulensi (hyaluronidase, collagenase). |
| **Benefits** | Target obat: inhibitor ACE (hipertensi), inhibitor protease HIV, inhibitor PDE5 (disfungsi ereksi). Digunakan dalam industri: enzim dalam deterjen, pengolahan makanan. |

#### 2 — Ion Channel

| Field | Konten |
|-------|--------|
| **Description** | Protein pori yang membentuk saluran di membran sel, memungkinkan ion (Na⁺, K⁺, Ca²⁺, Cl⁻) melewati membran. Dapat berupa voltage-gated, ligand-gated, atau mechanosensitive. |
| **Molecular Function** | Ion channel activity, voltage-gated ion channel activity, ligand-gated ion channel activity, ion transport |
| **Biological Process** | Action potential propagation, neurotransmission, muscle contraction, ion homeostasis, osmoregulation |
| **Cellular Location** | Cell membrane, plasma membrane, synaptic membrane, endoplasmic reticulum membrane |
| **Organisms** | Semua organisme — dari bakteri hingga manusia |
| **Function in Organisms** | Pada hewan: potensial aksi saraf, kontraksi otot. Pada tumbuhan: serapan nutrisi, respons stres. Pada bakteri: mekanoreseptor, homeostasis ion. |
| **Benefits** | Target anestesi lokal, antikonvulsan, antiaritmia, dan penghambat kanal kalsium. Mutasi kanal ion menyebabkan channelopathies (cystic fibrosis, epilepsy, cardiac arrhythmia). |

#### 3 — Kinase

| Field | Konten |
|-------|--------|
| **Description** | Enzim yang mengkatalisis transfer gugus fosfat dari ATP (atau GTP) ke substrat protein (fosforilasi). Mekanisme kunci dalam regulasi aktivitas protein dan transduksi sinyal. |
| **Molecular Function** | ATP binding, protein kinase activity, transferase activity, nucleotide binding |
| **Biological Process** | Protein phosphorylation, signal transduction, cell cycle regulation, apoptosis, cell differentiation, metabolism |
| **Cellular Location** | Cytoplasm, nucleus, cell membrane, mitochondrion |
| **Organisms** | Semua eukariota — manusia memiliki ~538 gen kinase (kinome). Juga ditemukan di prokariota (histidine kinase). |
| **Function in Organisms** | Pada manusia: mengatur pertumbuhan sel, sinyal insulin, respon imun. Pada tumbuhan: respons stres lingkungan, pertumbuhan. Pada ragi: kontrol siklus sel, mating response. |
| **Benefits** | Target utama obat kanker — >70 inhibitor kinase disetujui FDA (imatinib, gefitinib, erlotinib, sorafenib). Juga relevan untuk penyakit inflamasi dan neurodegeneratif. |

#### 4 — Oxidoreductase

| Field | Konten |
|-------|--------|
| **Description** | Enzim yang mengkatalisis reaksi oksidasi-reduksi (transfer elektron antara molekul). Termasuk dehidrogenase, oksidase, reduktase, peroksidase, dan hidroksilase. |
| **Molecular Function** | Oxidoreductase activity, electron transfer activity, NAD(P)H binding, FAD binding, heme binding |
| **Biological Process** | Cellular respiration, oxidative metabolism, redox homeostasis, detoxification, biosynthesis |
| **Cellular Location** | Mitochondrion, cytoplasm, peroxisome, endoplasmic reticulum, chloroplast (tumbuhan) |
| **Organisms** | Universal — ditemukan di semua organisme. Mitokondria pada eukariota adalah pusat reaksi oksidoreduktase. |
| **Function in Organisms** | Pada semua organisme: respirasi seluler (kompleks rantai transpor elektron), detoksifikasi (sitokrom P450 pada hati). Pada tumbuhan: fotosintesis, fiksasi nitrogen. |
| **Benefits** | Target obat: NSAID (siklooksigenase/COX), statin (HMG-CoA reduktase), antijamur azole (lanosterol 14α-demethylase). Digunakan dalam industri: biosensor, bioremediasi. |

#### 5 — Transcription Factor

| Field | Konten |
|-------|--------|
| **Description** | Protein yang mengikat sekuens DNA spesifik untuk mengaktivasi atau merepresi transkripsi gen. Mengandung DNA-binding domain (DBD) dan regulatory domain. Contoh: zinc finger, helix-turn-helix, leucine zipper. |
| **Molecular Function** | DNA binding, sequence-specific DNA binding, transcription regulatory region binding, transcription factor activity, RNA polymerase II binding |
| **Biological Process** | Regulation of transcription, gene expression, development, cell differentiation, response to stimulus, cell cycle |
| **Cellular Location** | Nucleus, nucleoplasm, chromatin |
| **Organisms** | Semua eukariota — manusia memiliki ~1.600 gen transcription factor. Juga ditemukan di prokariota. |
| **Function in Organisms** | Pada semua organisme: mengontrol gen mana yang diekspresikan. Pada manusia: perkembangan embrionik (Hox genes), respons hormon (steroid receptors), sirkadian rhythm (CLOCK/BMAL1). |
| **Benefits** | Target obat kanker (reseptor estrogen pada kanker payudara — tamoxifen), diabetes (PPARγ — thiazolidinediones), inflamasi (NF-κB — kortikosteroid). |

---

## 6. Implementasi Bertahap

### Tahap 1 — File Data (`class_info.json`)
- [ ] Buat file `data/class_info.json` dengan konten 6 famili
- [ ] Gunakan referensi biologis yang akurat (UniProt, Gene Ontology)

### Tahap 2 — Backend API
- [ ] Buat `api/main.py` — FastAPI app skeleton
- [ ] Buat `api/predict.py` — preprocessing + inference logic
- [ ] Buat `api/model_loader.py` — lazy load model (.pth)
- [ ] Endpoint `POST /predict` dengan validasi input
- [ ] Integrasikan `class_info.json` ke response

### Tahap 3 — Telegram Bot
- [ ] Setup `python-telegram-bot` dengan token
- [ ] Handler `/start` — pesan sambutan
- [ ] Handler `/help` — panduan lengkap
- [ ] Handler `/about` — info project + akurasi
- [ ] Handler `/clear` — hapus histori
- [ ] Handler `/compare` — side-by-side comparison
- [ ] Handler teks — deteksi sequence → prediksi
- [ ] Error handling + logging

### Tahap 4 — Testing & Deployment
- [ ] Test local dengan bot token development
- [ ] Test semua edge case (invalid sequence, long sequence, dll)
- [ ] Deploy API ke Railway / Hugging Face Spaces / VPS
- [ ] Deploy bot ke server (24/7)

---

## 7. Environment Variables (`.env`)

```env
BOT_TOKEN=your_telegram_bot_token_here
API_URL=http://localhost:8000
MODEL_DEFAULT=lstm
```

---

## 8. Dependencies Tambahan (`requirements.txt`)

```txt
# Tambahan untuk backend API
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.9.0

# Tambahan untuk Telegram Bot
python-telegram-bot==20.8
httpx==0.27.0
python-dotenv==1.0.0
```

---

## 9. Catatan Teknis

### Ukuran Pesan Telegram
- Telegram memiliki batas 4096 karakter per pesan.
- Output `/compare` (3 model + info class) mungkin perlu di-split jadi beberapa pesan.
- Solusi: info class hanya ditampilkan sekali di pesan pertama, perbandingan model di pesan kedua.

### Keamanan
- Token bot disimpan di `.env`, tidak di-commit ke git.
- Input sequence dibatasi 1000 karakter (validasi backend & bot).
- Rate limiting untuk mencegah spam.

### Biaya Operasional
- Model .pth + inference: ~50-200ms per prediksi (GPU) / ~1-3s (CPU).
- API bisa dijalankan di VPS sewa $5-10/bln atau free tier (Railway, Hugging Face Spaces).
- Bot Telegram gratis.
