# AGENTS.md — Protein Function Classification (LSTM)

## Project Type
Research project (Jupyter notebooks 01-04, Python inference in `api/`). Not pip-installable.

## Setup (Linux — Python 3.10)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name=protein-venv
cp .env.example .env   # Isi BOT_TOKEN dari @BotFather
```

## Key Commands
- `python run.py` — FastAPI (port 8000) + Telegram Bot (multiprocessing spawn)
- `python validate_nb.py` — validates notebook syntax (currently checks `04_lstm_v7_kaggle.ipynb`)

## Notebook Pipeline
01_data_acquisition → 02_preprocessing → 04_lstm_model (v5) / 04_lstm_v7_kaggle

All `04_lstm_*.ipynb` notebooks use Kaggle paths (`/kaggle/input/`, `/kaggle/working/`) — cannot re-run locally without changes.

## Model Artifacts (`models/`)
| Model | Weight File | Type |
|-------|-------------|------|
| LSTM (v5) | `lstm_model_best.pth` | state_dict |
| LSTM (v7) | `lstm_v7_best.pth` (Kaggle output) | state_dict |

`model_loader.py` loads from `models/lstm_model_best.pth`. The v5 checkpoint lives at `models/lstm_v5_output/models/lstm_model_best.pth` (symlink/copy needed for API). v7 checkpoint is in Kaggle working dir.

## Telegram Bot / API
- `.env` needs `BOT_TOKEN`, `API_URL`
- Model lazy-loaded with caching (`_model_cache` in `model_loader.py`)
- Sequence validation: `re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{4,1000}$')`
- API accepts only `lstm` model (since cleanup)

## Data (`data/` gitignored)
- `data/processed/{train,val,test}.csv`: [Entry, Sequence, Length, Label, Family]. Labels: 0=GPCR..5=TranscriptionFactor.
- `data/processed/label_mapping.json` — wajib, dibaca API & notebooks.
- `data/class_info.json` — hand-written educational data.
- `data/results/*_metrics.json`: `{accuracy, f1_macro, mcc, history}`

## Known Issues
- All LSTM notebooks hardcoded for Kaggle paths — must adapt paths for local re-run
- Live bot token leaked in git history (`planning/license_planning.md` on `main`)
- No CI, linting, type checking, or pre-commit configured
- `validate_nb.py` only checks `04_lstm_v7_kaggle.ipynb` — update path if notebook changes
