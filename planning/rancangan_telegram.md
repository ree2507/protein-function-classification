# Telegram Bot — Rancangan & Implementasi

## Arsitektur Sistem

```
User → Telegram Bot → FastAPI Backend → Model (.pth)
                    (python-telegram-bot)       ↓
                                         class_info.json
```

### Alur Prediksi
1. User kirim sequence protein (atau command)
2. Bot validasi input (regex `^[ACDEFGHIKLMNPQRSTVWY]{4,1000}$`)
3. Kirim POST `/predict` ke FastAPI backend
4. Backend: preprocessing → inference → format response + class_info
5. Bot: format Markdown → kirim ke user

## Implementasi

### Komponen

| File | Fungsi |
|------|--------|
| `api/main.py` | FastAPI — endpoint `/predict`, `/health`, `/models` |
| `api/predict.py` | Preprocessing + inference logic (LSTM) |
| `api/model_loader.py` | Lazy load model dengan caching (`_model_cache`) |
| `telegram_bot.py` | Bot handler — 3 command + auto-detect sequence |
| `run.py` | Launcher — spawn API + Bot via multiprocessing (spawn mode) |
| `data/class_info.json` | Edukasi 6 famili protein (179 baris) |
| `.env` | `BOT_TOKEN`, `API_URL`, `MODEL_DEFAULT` |

### Command Handlers

| Command | Deskripsi |
|---------|-----------|
| `/start` | Sambutan + daftar 6 famili + warning pengembangan |
| `/help` | Panduan lengkap penggunaan bot |
| `/about` | Info project, dataset, akurasi tiap model (HTML parse_mode) |
| **Input sequence** | Auto-detect → prediksi class + info edukasi |

### Key Decisions
- **Output:** Bahasa Indonesia
- **Low-confidence (<50%):** Saran validasi UniProt/NCBI
- **`/about`:** `parse_mode='HTML'` (untuk hindari error underscore)
- **`/clear`:** Dihapus
- **Message split:** Output >4096 karakter dipecah (batas Telegram)

### Output Edukatif
Setiap prediksi menampilkan: Deskripsi famili, Fungsi Molekuler (GO), Proses Biologis, Lokasi Sel, Organisme umum, Rincian Probabilitas per kelas, Peringatan jika confidence <50%.

### Cara Menjalankan
```bash
pip install -r requirements.txt
# Isi BOT_TOKEN di .env (dari @BotFather)
python run.py
```

### Catatan Teknis
- Bot berjalan lokal via `python run.py` (API port 8000)
- Model lazy-loaded dengan caching — stay in memory setelah first load
- `run.py` menggunakan `multiprocessing.set_start_method('spawn')` (wajib di Windows)
- `.env` tidak di-commit (ada di `.gitignore`)

## Rencana Selanjutnya
- Deployment 24/7 ke Railway / Hugging Face Spaces (free tier)
- Fitur: inline button untuk quick model switch
- Fitur: conversation handler untuk pengalaman interaktif
- Rate limiting untuk mencegah spam
