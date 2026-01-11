import google.generativeai as genai
from app.config import settings

def analyze_market_with_gemini(
    symbol: str,
    timeframe: str,
    bias: str,
    close: float,
    ema50: float,
    ema200: float,
    rsi: float,
    zones: dict,
    liquidity: dict
) -> str:
    """
    Kirim data teknikal ke Gemini untuk dapat analisa naratif ala Institutional Trader.
    """
    if not settings.use_gemini or not settings.gemini_api_key:
        return "AI Analysis Disabled."

    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        # Contextual Prompt
        liq_sweep = liquidity.get("sweep", "None")
        liq_note = liquidity.get("notes", "-")
        
        demand = zones.get("DEMAND")
        supply = zones.get("SUPPLY")
        
        zone_info = "No nearby zones."
        if demand:
            zone_info += f" Demand Zone at {demand['low']:.2f}-{demand['high']:.2f}."
        if supply:
            zone_info += f" Supply Zone at {supply['low']:.2f}-{supply['high']:.2f}."

        prompt = f"""
        Role: You are a Senior Institutional Trader & Technical Analyst (SMC Strategy).
        Task: Analyze the current market setup for {symbol} on {timeframe} timeframe based on the data below.

        Technical Data:
        - Current Price: {close}
        - Trend Bias: {bias}
        - EMA Structure: EMA50 is {'ABOVE' if ema50 > ema200 else 'BELOW'} EMA200. Price is {'ABOVE' if close > ema50 else 'BELOW'} EMA50.
        - RSI (14): {rsi:.2f}
        - Key Zones: {zone_info}
        - Liquidity Event: {liq_sweep} ({liq_note})

        Output Requirement:
        1. Write a SHORT, PUNCHY analysis in BAHASA INDONESIA.
        2. Focus on: Why the bias is {bias}, logic of zones/liquidity, and potential scenario.
        3. Style: Professional but relaxed ("Bro", "Trader"). No hedging/disclaimer needed.
        4. Max Length: 2-3 short paragraphs.
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return f"AI Analysis Failed: {str(e)}"