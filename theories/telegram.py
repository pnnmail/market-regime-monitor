import os

import requests

# Telegram accepts up to 4096 characters per message; keep a margin
MAX_LEN = 3900


def split_message(message, limit=MAX_LEN):
    """Splits text into parts no longer than limit along line boundaries (a long line is cut hard)."""
    parts, cur = [], ""
    for line in message.split("\n"):
        while len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts or [""]


def send_message(message):

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:

        print(
            "Telegram disabled: "
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
            "is missing."
        )

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    parts = split_message(message)
    ok = True

    for i, part in enumerate(parts, 1):

        try:

            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )

            response.raise_for_status()

            print(
                "Telegram: message sent."
                if len(parts) == 1
                else f"Telegram: part {i}/{len(parts)} sent."
            )

        except Exception as exc:

            # the request URL contains the bot token — do not write it to the log
            print(
                f"Telegram error (part {i}/{len(parts)}): "
                f"{str(exc).replace(token, '<token hidden>')}"
            )

            ok = False

    return ok
