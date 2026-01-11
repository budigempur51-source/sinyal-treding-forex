from typing import Dict, Tuple

def no_trade_gate(
    tf_results: Dict[str, Dict],
    min_atr_map: Dict[str, float],
) -> Tuple[bool, str]:
    """
    Decide whether market is tradable.
    Returns:
      (allowed: bool, reason: str)

    tf_results[tf] expects:
      - bias
      - event
      - atr
      - vol_z
    """

    # 1) HTF alignment (H1 & H4 must agree)
    h1 = tf_results.get("H1")
    h4 = tf_results.get("H4")

    if not h1 or not h4:
        return False, "HTF data missing"

    if h1["bias"] != h4["bias"]:
        return False, f"HTF conflict (H1={h1['bias']} vs H4={h4['bias']})"

    if h1["bias"] == "RANGING":
        return False, "HTF ranging"

    # 2) CHoCH on HTF = unstable
    if h1["event"] == "CHoCH" or h4["event"] == "CHoCH":
        return False, "HTF CHoCH detected"

    # 3) ATR filter (avoid chop)
    for tf, min_atr in min_atr_map.items():
        tf_data = tf_results.get(tf)
        if not tf_data:
            continue
        if tf_data["atr"] < min_atr:
            return False, f"Low ATR on {tf}"

    # 4) Volume sanity (tick volume z-score)
    # Too negative = dead market
    for tf in ["M15", "H1"]:
        tf_data = tf_results.get(tf)
        if not tf_data:
            continue
        if tf_data["vol_z"] < -3.0:
            return False, f"Dead volume on {tf}"

    return True, "Market OK"
