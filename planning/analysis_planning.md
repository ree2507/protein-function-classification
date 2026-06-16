# Comparative Analysis — [DEPRECATED / Historical]

Dokumen ini adalah hasil analisis komparatif ketika project masih menggunakan 3 model (CNN, LSTM, ESM-2). CNN dan ESM-2 telah dihapus; project sekarang fokus ke LSTM saja. Data LSTM di bawah tetap relevan sebagai baseline.

### Metrik Final

| Metrik | LSTM v5 |
|--------|:-------:|
| **Accuracy** | 86.85% |
| **F1 Macro** | 0.8716 |
| **MCC** | 0.8422 |
| **Precision Macro** | 0.8726 |
| **Recall Macro** | 0.8720 |

### F1-Score per Kelas

| Kelas | LSTM v5 |
|-------|:-------:|
| GPCR | 0.9594 |
| Hydrolase | 0.7456 |
| Ion Channel | 0.9559 |
| Kinase | 0.8703 |
| Oxidoreductase | 0.8340 |
| Transcription Factor | 0.8647 |

### Waktu Inference (3,764 sekuen, RTX 2050)

| Model | Waktu |
|-------|:-----:|
| LSTM v5 | 4.1s |

## Catatan

- **Hydrolase adalah kelas tersulit** — F1 0.75, diversitas sekuens tertinggi.
- Detail optimasi LSTM: [`planning/lstm_planning.md`](planning/lstm_planning.md).
