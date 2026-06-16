# License Planning

Dokumen ini adalah *working document* untuk direview kembali saat project 100% siap dipublikasikan. Berisi status lisensi semua komponen yang digunakan dan hal-hal yang perlu diputuskan.

---

## 1. Status Lisensi Komponen

### Libraries & Frameworks

| Komponen | Lisensi | Syarat |
|----------|---------|--------|
| PyTorch | BSD | Redistribusi source/binary wajib sertakan copyright notice asli. Tidak boleh pakai nama "Facebook/PyTorch" untuk endorse produk turunan. |
| Transformers (HuggingFace) | Apache 2.0 | Attribution wajib. |
| PEFT | Apache 2.0 | Attribution wajib. |
| Accelerate | Apache 2.0 | Attribution wajib. |
| FastAPI | MIT | Attribution wajib. |
| Uvicorn | BSD | Attribution wajib. |
| scikit-learn | BSD | Attribution wajib. |
| pandas | BSD | Attribution wajib. |
| numpy | BSD | Attribution wajib. |
| matplotlib | BSD | Attribution wajib. |
| seaborn | BSD | Attribution wajib. |
| Biopython | Biopython License | BSD-like, attribution. |
| httpx | BSD | Attribution wajib. |
| python-dotenv | BSD | Attribution wajib. |
| Jupyter | BSD | Attribution wajib. |
| python-telegram-bot | LGPLv3 | **Tidak memengaruhi lisensi project.** LGPL hanya berlaku untuk library itu sendiri — aplikasi yang menggunakannya tidak wajib GPL. |

### Dataset

| Dataset | Lisensi | Syarat |
|---------|---------|--------|
| UniProt Swiss-Prot | **CC BY 4.0** | Boleh direproduksi, dimodifikasi, dan didistribusikan ulang (termasuk komersial) selama atribusi diberikan. **Ini satu-satunya komponen dengan kewajiban atribusi eksplisit.** |

### Model dari Scratch (Milik Anda)

| Model | Status Hak Cipta |
|-------|-----------------|
| LSTM v5 (`ProteinLSTM`) | Milik Anda — implementasi sendiri |
| Transformer (direncanakan) | Milik Anda — implementasi sendiri |

---

## 2. Model Naming

Tidak ada batasan hukum untuk memberi nama model dari scratch dengan nama apa pun:
- Model dari scratch (LSTM, Transformer) adalah implementasi orisinal → **hak cipta Anda**
- Tidak ada klaim atas nama arsitektur publik (LSTM, Transformer) karena itu terminologi ilmiah

---

## 3. Checklist Publikasi

Centang sebelum project dijadikan public / open source:

- [ ] **Hapus `.env` dari riwayat git** — mengandung live Telegram bot token (`8613279756:AAGee1kdQ8St3MZRrDeLp9OcvjWBdPvT1n8`). Gunakan `git filter-branch` atau BFG Repo-Cleaner.
- [ ] **Tambah file `LICENSE`** — pilih lisensi untuk project (lihat bagian 4).
- [ ] **Tambah atribusi UniProt** di `README.md` — contoh:
  > *Dataset protein bersumber dari UniProt/Swiss-Prot (CC BY 4.0). https://www.uniprot.org/*
- [ ] **Tambah atribusi framework utama** di `README.md` atau `NOTICE` — cukup daftar singkat.
- [ ] **Cek ulang semua file konfigurasi** — pastikan tidak ada token/kunci lain yang terekspos.
- [ ] **Tentukan versi final model weights** — file `.pth` di `models/` perlu di-ignore atau didistribusikan terpisah jika ukurannya besar.

---

## 4. Pilihan Lisensi Project (TBD — Putuskan Nanti)

| Lisensi | Open Source? | Wajib copyleft? | Cocok untuk |
|---------|:-----------:|:----------------:|-------------|
| **MIT** | ✅ Sangat bebas | Tidak | Paling sederhana, izinkan penggunaan apa pun |
| **Apache 2.0** | ✅ Bebas + paten | Tidak | Mirip MIT ada klausa paten eksplisit |
| **GPLv3** | ✅ Paling ketat | Ya, turunan harus GPL | Memaksa kontribusi balik ke komunitas |
| **CC BY 4.0** | ✅ (untuk dataset/konten) | Tidak | Hanya untuk dataset/dokumentasi, bukan kode |

**Catatan:** Semua lisensi di atas kompatibel dengan komponen yang digunakan. Tidak ada komponen yang menggunakan GPL "strong" yang bisa memaksa project ikut GPL.

---

## 5. Ringkasan

- ✅ Tidak ada hambatan hukum untuk open source
- ✅ Tidak ada hambatan untuk menamai model sendiri
- ✅ Semua lisensi framework bersifat permissive
- ⚠️ Satu kewajiban atribusi: **UniProt (CC BY 4.0)**
- ⚠️ Satu risiko keamanan: **live token di riwayat git `.env`** — harus dibersihkan
- ❓ Yang perlu diputuskan: lisensi project (MIT / Apache 2.0 / GPL / lain)
