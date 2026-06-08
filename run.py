import os
import sys
import time
import multiprocessing
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

WAIT_FOR_API = 5
MAX_RETRIES = 12


def run_api():
    import uvicorn
    logger.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_level="info")


def wait_for_api():
    import httpx
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = httpx.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                logger.info("API server is ready.")
                return True
        except Exception:
            pass
        logger.info(f"Waiting for API server... (attempt {attempt}/{MAX_RETRIES})")
        time.sleep(WAIT_FOR_API / MAX_RETRIES)
    logger.error("API server did not start in time.")
    return False


def run_bot():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv
    load_dotenv()
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token or bot_token == "your_telegram_bot_token_here":
        logger.error("BOT_TOKEN not set in .env file! Bot will not start.")
        sys.exit(1)
    import telegram_bot
    logger.info("Starting Telegram bot...")
    telegram_bot.main()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    api_process = multiprocessing.Process(target=run_api, daemon=True)
    bot_process = None

    api_process.start()
    logger.info(f"API process started, waiting up to {WAIT_FOR_API}s for it to be ready...")

    if not wait_for_api():
        logger.error("Could not start API server. Exiting.")
        api_process.terminate()
        sys.exit(1)

    bot_process = multiprocessing.Process(target=run_bot, daemon=True)
    bot_process.start()
    logger.info("Bot process started.")

    print("\n" + "=" * 60)
    print("  🧬 Protein Classifier — Running")
    print("  API:  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Bot:  Active (polling)")
    print("=" * 60)
    print("  Press Ctrl+C to stop all services.\n")

    try:
        while True:
            time.sleep(1)
            if not api_process.is_alive():
                logger.error("API process died unexpectedly.")
                break
            if bot_process and not bot_process.is_alive():
                logger.error("Bot process died unexpectedly.")
                break
    except KeyboardInterrupt:
        logger.info("\nShutting down all services...")
    finally:
        if api_process and api_process.is_alive():
            api_process.terminate()
            api_process.join()
        if bot_process and bot_process.is_alive():
            bot_process.terminate()
            bot_process.join()
        logger.info("All services stopped.")
