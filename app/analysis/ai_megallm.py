from openai import OpenAI
from app.config import settings

def analyze_market_with_megallm(
    symbol: str,
    timeframe: str,
    bias: str,
    close: float,
    ema50: float,
    ema200: float,
    rsi: float,
    zones: dict,
    liquidity: dict,
    context: str = "TRADE" # TRADE atau WATCH
) -> str:
    """
    Kirim data teknikal ke MegaLLM (DeepSeek/OpenAI) untuk analisa naratif Institutional.
    """
    if not settings.use_ai_narrative or not settings.megallm_api_key:
        return "AI Analysis Disabled."

    try:
        # Init Client (MegaLLM)
        client = OpenAI(
            api_key=settings.megallm_api_key,
            base_url=settings.megallm_base_url
        )

        # Contextual Prompt Construction
        liq_sweep = liquidity.get("sweep", "None")
        liq_note = liquidity.get("notes", "-")
        fakeout = liquidity.get("fake", "None") 
        
        demand = zones.get("DEMAND")
        supply = zones.get("SUPPLY")
        
        zone_info = "Tidak ada zona terdekat."
        if demand:
            zone_info += f" DEMAND di {demand['low']:.2f}-{demand['high']:.2f}."
        if supply:
            zone_info += f" SUPPLY di {supply['low']:.2f}-{supply['high']:.2f}."

        # Prompt Engineer: Institutional Style with SCENARIOS
        prompt = f"""
        Role: Anda adalah Senior Analyst & Trader Institusional (SMC Specialist). 
        Tugas: Analisa setup market {symbol} (TF {timeframe}) untuk komunitas trader Indonesia.

        Data Teknikal:
        - Harga Saat Ini: {close}
        - Bias Struktur: {bias}
        - Tren EMA: EMA50 {'DI ATAS' if ema50 > ema200 else 'DI BAWAH'} EMA200. Harga {'DI ATAS' if close > ema50 else 'DI BAWAH'} EMA50.
        - RSI (14): {rsi:.2f}
        - Zona Kunci: {zone_info}
        - Likuiditas: Sweep={liq_sweep}, Fakeout={fakeout}
        - Konteks Laporan: {context} (Jika WATCH = Belum ada entry valid. Jika TRADE = Ada sinyal valid).

        Instruksi Output (Bahasa Indonesia Tegas & Santai):
        1. **Analisa Struktur**: Jelaskan kenapa bias {bias} valid atau lemah berdasarkan EMA & Struktur.
        2. **Skenario Pergerakan**: 
           - "Jika harga break [Level Supply/Demand]..."
           - "Jika harga reject di [Level]..."
        3. **Rekomendasi**: 
           - Kalau {context} == 'TRADE': Validasi alasan entrynya.
           - Kalau {context} == 'WATCH': Kasih saran "Tunggu apa?" (misal: Tunggu Sweep low dulu).
        
        Format: Gunakan bullet points atau paragraf pendek. Jangan pakai disclaimer klise. Fokus ke "Actionable Insight".
        """

        response = client.chat.completions.create(
            model=settings.megallm_model,
            messages=[
                {"role": "system", "content": "You are a professional crypto/forex analyst relying on Price Action & SMC."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=450
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ AI/MegaLLM Error: {e}")
        return f"AI Analysis Failed: {str(e)}"