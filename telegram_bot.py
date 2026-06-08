import os
import re
import logging
from dotenv import load_dotenv
import httpx

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('BOT_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000')
MODEL_DEFAULT = os.getenv('MODEL_DEFAULT', 'lstm')

SEQ_PATTERN = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{4,1000}$')

CLASS_EMOJIS = {
    "GPCR": "📡",
    "Hydrolase": "💧",
    "Ion Channel": "⚡",
    "Kinase": "🔋",
    "Oxidoreductase": "🔄",
    "Transcription Factor": "🧬"
}

MODEL_NAMES = {
    "cnn": "CNN v2",
    "lstm": "LSTM v5",
    "esm2": "ESM-2 v1"
}

def build_prediction_text(result, model_key="lstm"):
    if model_key == "all":
        text = "🧬 *Perbandingan Semua Model*\n"
        text += f"Panjang sekuens: {result['sequence_length']} aa\n\n"
        for mkey, mname in [("cnn", "CNN v2"), ("lstm", "LSTM v5"), ("esm2", "ESM-2 v1")]:
            pred = result['predictions'][mkey]
            emoji = CLASS_EMOJIS.get(pred['class'], "🔬")
            text += f"*{mname}:* {emoji} {pred['class']} ({pred['confidence']:.1%})\n"
        info = result['class_info']
        text += f"\n📖 *{info['name_full']}*\n{info['description']}"
        return text

    pred = result['prediction']
    info = result['class_info']
    emoji = CLASS_EMOJIS.get(pred['class'], "🔬")

    text = f"🧬 *Hasil Prediksi*\n"
    text += f"Panjang sekuens: `{result['sequence_length']} aa`\n"
    text += f"Model: {MODEL_NAMES.get(result['model_used'], result['model_used'])}\n\n"

    text += f"🔮 *Prediksi:* {emoji} `{pred['class']}` ({pred['confidence']:.1%})\n\n"

    text += f"📖 *Deskripsi*\n{info.get('description', 'N/A')}\n\n"

    mf = info.get('molecular_function', [])
    if mf:
        text += "⚙️ *Fungsi Molekuler*\n"
        for item in (mf if isinstance(mf, list) else [mf]):
            text += f"• {item}\n"
        text += "\n"

    bp = info.get('biological_process', [])
    if bp:
        text += "🔬 *Proses Biologis*\n"
        for item in (bp if isinstance(bp, list) else [bp]):
            text += f"• {item}\n"
        text += "\n"

    cl = info.get('cellular_location', [])
    if cl:
        text += "📍 *Lokasi Sel*\n"
        for item in (cl if isinstance(cl, list) else [cl]):
            text += f"• {item}\n"
        text += "\n"

    org = info.get('organisms', [])
    if org:
        text += "🧫 *Organisme Umum*\n"
        for item in (org if isinstance(org, list) else [org]):
            text += f"• {item}\n"
        text += "\n"

    text += "📊 *Rincian Probabilitas:*\n"
    probs = pred['probabilities']
    for cls_name, prob in probs.items():
        cls_emoji = CLASS_EMOJIS.get(cls_name, "")
        if cls_name == pred['class']:
            text += f"├─ {cls_emoji} {cls_name:<20s} {prob:.1%} ✅\n"
        else:
            text += f"├─ {cls_emoji} {cls_name:<20s} {prob:.1%}\n"

    if pred['confidence'] < 0.5:
        text += (
            "\n\n⚠️ *Hasil prediksi ini memiliki probabilitas rendah.*\n"
            "Disarankan untuk memvalidasi kembali sekuens protein "
            "Anda di sumber terpercaya seperti:\n"
            "• UniProt (www.uniprot.org)\n"
            "• NCBI BLAST (blast.ncbi.nlm.nih.gov)"
        )

    return text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        # "╔══════════════════════════════════╗\n"
        "   🧬 *Protein Classifier Bot*    \n"
        # "╚══════════════════════════════════╝\n\n"
        "Selamat datang! Saya adalah bot untuk mengklasifikasikan\n"
        "fungsi protein berdasarkan sekuens asam aminonya.\n\n"
        "🔬 *Model Tersedia:*\n"
        "• CNN (Conv1D) — 82.76% akurasi\n"
        "• Bi-LSTM — 86.85% akurasi ✅ (default)\n"
        "• ESM-2 — 93.09% akurasi\n\n"
        "🧪 *6 Famili Protein:*\n"
        "0️⃣ GPCR — G Protein-Coupled Receptor\n"
        "1️⃣ Hydrolase — Enzim hidrolisis\n"
        "2️⃣ Ion Channel — Saluran ion membran\n"
        "3️⃣ Kinase — Enzim fosforilasi\n"
        "4️⃣ Oxidoreductase — Enzim redoks\n"
        "5️⃣ Transcription Factor — Regulasi transkripsi\n\n"
        "📝 *Cara Pakai:*\n"
        "Kirimkan sekuens protein (huruf A-Z, 4-1000 karakter):\n"
        "Contoh: `MKTAYIA...`\n\n"
        "📋 *Daftar Perintah:*\n"
        "/start — Tampilkan pesan ini\n"
        "/help — Panduan lengkap\n"
        "/about — Info project & akurasi\n"
        "/compare <seq> — Bandingkan semua model\n"
        "/model <cnn/lstm/esm2> — Ganti model default\n\n"
        "⚠️ *Warning*\n"
        "Project ini masih dalam tahap pengembangan, sehingga\n"
        "informasi yang disajikan serta hasil prediksi model\n"
        "kemungkinan masih kurang akurat."
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📋 *Panduan Lengkap*\n\n"
        "🔹 *Prediksi Sekuens:*\n"
        "Kirim teks yang berisi sekuens protein\n"
        "(hanya huruf A-Z, panjang 4-1000 karakter).\n\n"
        "🔹 *Ganti Model:*\n"
        "Gunakan /model diikuti nama model:\n"
        "• `/model cnn` — CNN v2 (cepat, 82.76% akurasi)\n"
        "• `/model lstm` — LSTM v5 (sedang, 86.85% akurasi)\n"
        "• `/model esm2` — ESM-2 (lambat, 93.09% akurasi)\n\n"
        "🔹 *Bandingkan:*\n"
        "`/compare MKTAYIA...` — Prediksi dari 3 model\n"
        "sekaligus (side-by-side).\n\n"
        "🔹 *Model Default:*\n"
        f"Saat ini: `{MODEL_DEFAULT}`\n"
        "Ganti dengan /model\n\n"
        "🔹 *Batasan:*\n"
        "• Maks 1000 karakter per sekuens\n"
        "• Hanya 20 asam amino standar\n"
        "  (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🧬 <b>Tentang Project</b>\n\n"
        "📌 <b>Deskripsi</b>\n"
        "Project perbandingan arsitektur deep learning untuk\n"
        "klasifikasi fungsi protein.\n\n"
        "🎯 <b>Tujuan</b>\n"
        "Sebagai bahan pembelajaran dan edukasi mengenai\n"
        "penerapan deep learning dalam bioinformatika.\n\n"
        "💡 <b>Manfaat</b>\n"
        "Mengenal tiga pendekatan AI (CNN, LSTM, ESM-2)\n"
        "dalam klasifikasi fungsi protein secara langsung.\n\n"
        "📊 <b>Deskripsi Dataset</b>\n"
        "• Sumber: UniProt Swiss-Prot\n"
        "• Total: ~25.000 sekuens protein\n"
        "• Split: 70% Train, 15% Validasi, 15% Test\n"
        "• 6 famili: GPCR, Hydrolase, Ion Channel,\n"
        "  Kinase, Oxidoreductase, Transcription Factor\n\n"
        "👨‍💻 <b>Info Pembuat</b>\n"
        "made by atmin\n"
        "• Instagram: https://www.instagram.com/reyhandny_\n"
        "• LinkedIn: https://www.linkedin.com/in/reyhandany\n"
        "• GitHub: https://github.com/ree2507"
    )
    await update.message.reply_text(text, parse_mode='HTML')

async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    seq = ' '.join(context.args)
    if not seq:
        await update.message.reply_text(
            "⚠️ Gunakan: `/compare MKTAYIA...`\n"
            "Contoh: `/compare MKTAYIAQQLQ`",
            parse_mode='Markdown'
        )
        return
    seq = seq.upper()
    if not SEQ_PATTERN.match(seq):
        await update.message.reply_text(
            "❌ Sekuens tidak valid. Hanya huruf A-Z, panjang 4-1000.",
            parse_mode='Markdown'
        )
        return
    await update.message.reply_text("⏳ Membandingkan semua model... (ESM-2 mungkin perlu ~30 detik)")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{API_URL}/predict", json={"sequence": seq, "model": "all"})
            if r.status_code != 200:
                await update.message.reply_text(f"❌ Error API: {r.json().get('detail', 'Unknown error')}")
                return
            result = r.json()
        text = build_prediction_text(result, "all")
        if len(text) > 4000:
            parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for part in parts:
                await update.message.reply_text(part, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal menghubungi API: {e}")

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            f"Model saat ini: `{MODEL_DEFAULT}`\n"
            "Ganti dengan: `/model cnn`, `/model lstm`, atau `/model esm2`",
            parse_mode='Markdown'
        )
        return
    model = context.args[0].lower()
    if model not in ("cnn", "lstm", "esm2"):
        await update.message.reply_text(
            "❌ Model tidak dikenal. Pilih: `cnn`, `lstm`, atau `esm2`",
            parse_mode='Markdown'
        )
        return
    context.user_data['model'] = model
    await update.message.reply_text(
        f"✅ Model default diubah ke: *{MODEL_NAMES[model]}*",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not SEQ_PATTERN.match(text):
        await update.message.reply_text(
            "❕ Itu bukan sekuens protein yang valid.\n\n"
            "Kirim teks berisi huruf A-Z (panjang 4-1000) untuk prediksi,\n"
            "atau gunakan /help untuk panduan.",
            parse_mode='Markdown'
        )
        return
    seq = text.upper()
    model = context.user_data.get('model', MODEL_DEFAULT)
    model_label = MODEL_NAMES.get(model, model)
    await update.message.reply_text(f"⏳ Memprediksi dengan *{model_label}*...", parse_mode='Markdown')
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{API_URL}/predict", json={"sequence": seq, "model": model})
            if r.status_code != 200:
                await update.message.reply_text(f"❌ Error API: {r.json().get('detail', 'Unknown error')}")
                return
            result = r.json()
        text = build_prediction_text(result)
        if len(text) > 4000:
            parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for part in parts:
                await update.message.reply_text(part, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal menghubungi API: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set in .env file!")
        return
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("model", set_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    logger.info("Bot started polling...")
    app.run_polling()

if __name__ == '__main__':
    main()
