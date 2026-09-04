import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier

# ================= AYARLAR =================
API_KEY = "LCRYKTbUeLyEVi7EUCHid7f0n7iRowyLb90PEqGve6pdEKUuuY62RjrQbQfnau8o"
API_SECRET = "ThaB1p140sREEqMsu32g62N7pYoSLp5nXLFCgebncpa76u0dR76IF8JXu1PvFDA3"

INTERVAL = Client.KLINE_INTERVAL_15MINUTE  # 15 Dakikalık mumlar
BUDGET_LIMIT_USDT = 20.0                   # 20$ Bütçe
TOP_COINS_COUNT = 20                       # En hacimli 20 coin
LOOP_INTERVAL_SECONDS = 180                # Her 180 saniyede (3 dakikada) bir piyasayı kontrol et

STATE_FILE = "bot_state.json"
LOG_FILE = "trade_log.txt"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"in_position": False, "active_symbol": None, "buy_price": 0.0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def safe_binance_client():
    for attempt in range(5):
        try:
            client = Client()
            return client
        except Exception:
            time.sleep(3)
    return None

def get_top_volume_usdt_pairs(client, limit=20):
    tickers = client.get_ticker()
    usdt_pairs = [
        t for t in tickers 
        if t['symbol'].endswith('USDT') 
        and not t['symbol'].startswith('USDC') 
        and not t['symbol'].startswith('FDUSD')
    ]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [p['symbol'] for p in sorted_pairs[:limit]]

def build_features(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA_Diff"] = (df["SMA20"] - df["SMA50"]) / df["Close"]
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(10).std()
    
    df["Future_Return"] = df["Close"].shift(-4) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0.010).astype(int)
    
    return df.dropna()

def train_and_predict(df):
    features = ["SMA_Diff", "RSI", "Volatility", "Returns"]
    X = df[features].iloc[:-1]
    y = df["Target"].iloc[:-1]
    X_latest = df[features].iloc[[-1]]
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    prob = model.predict_proba(X_latest)[0][1]
    return prob

def run_bot_cycle():
    public_client = safe_binance_client()
    if not public_client:
        log("Binance bağlantısı sağlanamadı, bir sonraki döngüde tekrar denenecek.")
        return

    state = load_state()
    
    # 1. POZİSYONDAYSAK KONTROL ET
    if state["in_position"]:
        symbol = state["active_symbol"]
        buy_price = state["buy_price"]
        
        klines = public_client.get_klines(symbol=symbol, interval=INTERVAL, limit=10)
        current_price = float(klines[-1][4])
        change_pct = (current_price - buy_price) / buy_price
        
        log(f"TAKİP: {symbol} | Alış: {buy_price} | Güncel: {current_price} | Kâr/Zarar: %{change_pct*100:.2f}")
        
        # Stop-Loss (-%1.0) veya Take-Profit (+%2.0)
        if change_pct <= -0.010 or change_pct >= 0.020:
            log(f"--- SAT SİNYALİ ({symbol})! Kâr/Zarar: %{change_pct*100:.2f} ---")
            state["in_position"] = False
            state["active_symbol"] = None
            state["buy_price"] = 0.0
            save_state(state)
            log("Pozisyon kapatıldı (Nakit duruma geçildi).")
        return

    # 2. PİYASA TARAMASI (20 COIN)
    log("=== PİYASA TARAMASI BAŞLADI (En Hacimli 20 Coin) ===")
    top_symbols = get_top_volume_usdt_pairs(public_client, limit=TOP_COINS_COUNT)
    
    best_symbol = None
    best_prob = 0.0
    best_price = 0.0
    
    for symbol in top_symbols:
        try:
            klines = public_client.get_klines(symbol=symbol, interval=INTERVAL, limit=200)
            data = [{"Close": float(k[4]), "High": float(k[2]), "Low": float(k[3]), "Volume": float(k[5])} for k in klines]
            df = pd.DataFrame(data)
            df_prepared = build_features(df)
            
            price = df["Close"].iloc[-1]
            prob = train_and_predict(df_prepared)
            
            if prob > best_prob:
                best_prob = prob
                best_symbol = symbol
                best_price = price
        except Exception:
            continue
            
    log(f"En Yüksek Fırsat: {best_symbol} (AI Yükseliş İhtimali: %{best_prob*100:.1f})")
    
    if best_prob >= 0.55:
        log(f"--- GÜÇLÜ SİNYAL! {best_symbol} alınıyor... ---")
        state["in_position"] = True
        state["active_symbol"] = best_symbol
        state["buy_price"] = best_price
        save_state(state)
        log(f"AL Emri Başarılı: {best_symbol} - Alış Fiyatı: {best_price}")
    else:
        log("Yeterli alım fırsatı yok, bekleniyor.")

# SONSUR DÖNGÜ (Kendi Kendine Çalışan Yapı)
if __name__ == "__main__":
    log("Kripto Yapay Zeka Botu 7/24 Kesintisiz Moda Geçti.")
    while True:
        try:
            run_bot_cycle()
        except Exception as e:
            log(f"Döngü hatası: {e}")
            
        # 3 dakika bekle ve tekrar çalıştır
        time.sleep(LOOP_INTERVAL_SECONDS)