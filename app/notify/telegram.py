import requests
from app.config import settings


def send_message(text: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("⚠️ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        print("❌ Telegram send failed:", r.text)


def send_photo(caption: str, image_path: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("⚠️ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendPhoto"
    with open(image_path, "rb") as f:
        files = {"photo": f}
        data = {
            "chat_id": settings.telegram_chat_id,
            "caption": caption,
            "parse_mode": "Markdown",
        }
        r = requests.post(url, files=files, data=data, timeout=20)
        if r.status_code != 200:
            print("❌ Telegram photo failed:", r.text)
