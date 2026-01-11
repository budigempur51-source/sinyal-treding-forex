import aiohttp
import json
from typing import Optional

from app.config import settings

DISCORD_API = "https://discord.com/api/v10"


def _auth_headers():
    return {
        "Authorization": f"Bot {settings.discord_bot_token}",
        "User-Agent": "xau-signal-engine/1.0",
    }


async def send_discord_embed(
    title: str,
    description: str,
    color: int = 0x2ECC71,
    footer: Optional[str] = None,
):
    """
    Send embed-only message to a Discord channel.
    """
    url = f"{DISCORD_API}/channels/{settings.discord_channel_id}/messages"

    embed = {
        "title": title,
        "description": description,
        "color": color,
    }
    if footer:
        embed["footer"] = {"text": footer}

    payload = {"embeds": [embed]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers={**_auth_headers(), "Content-Type": "application/json"}, json=payload) as r:
            text = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"Discord send failed ({r.status}): {text}")


async def send_discord_embed_with_image(
    title: str,
    description: str,
    image_path: str,
    color: int = 0x3498DB,
    footer: Optional[str] = None,
):
    """
    Send embed + attached image (PNG) to a Discord channel.
    Uses multipart form-data (file + payload_json).
    """
    url = f"{DISCORD_API}/channels/{settings.discord_channel_id}/messages"

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "image": {"url": "attachment://chart.png"},
    }
    if footer:
        embed["footer"] = {"text": footer}

    payload = {"embeds": [embed]}

    form = aiohttp.FormData()
    form.add_field("payload_json", json.dumps(payload), content_type="application/json")

    with open(image_path, "rb") as f:
        form.add_field(
            "files[0]",
            f,
            filename="chart.png",
            content_type="image/png",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=_auth_headers(), data=form) as r:
                text = await r.text()
                if r.status >= 300:
                    raise RuntimeError(f"Discord send failed ({r.status}): {text}")
