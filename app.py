# === Indicator Functions ===

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, period=20):
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower

# === API Endpoint ===

@app.route('/api/stock', methods=['GET'])
def analyze_stock():
    symbol = request.args.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': 'Missing stock symbol'}), 400

    try:
        df = yf.download(symbol, period='3mo', interval='1d')
        if df.empty:
            return jsonify({'error': f"No data found for {symbol}"}), 404

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        close = df['Close']

        sma20 = calculate_sma(close, 20).iloc[-1]
        sma50 = calculate_sma(close, 50).iloc[-1]
        ema20 = calculate_ema(close, 20).iloc[-1]
        rsi = calculate_rsi(close).iloc[-1]
        macd, signal, histogram = calculate_macd(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)

        weekly_avg = df['Close'].resample('W').mean().dropna().round(2).tail(4).to_dict()
        weekly_avg = {str(date.date()): price for date, price in weekly_avg.items()}

        response = {
            "symbol": symbol,
            "latestPrice": round(close.iloc[-1], 2),
            "change": round(close.iloc[-1] - close.iloc[-2], 2),
            "changePercent": round((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2),
            "volume": int(df['Volume'].iloc[-1]),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "ema20": round(ema20, 2),
            "rsi": round(rsi, 2),
            "macd": round(macd.iloc[-1], 4),
            "macdSignal": round(signal.iloc[-1], 4),
            "macdHistogram": round(histogram.iloc[-1], 4),
            "bollingerBands": {
                "upper": round(bb_upper.iloc[-1], 2),
                "middle": round(bb_middle.iloc[-1], 2),
                "lower": round(bb_lower.iloc[-1], 2),
            },
            "weeklyAveragePrices": weekly_avg
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
@app.route('/api/stock', methods=['GET'])
def analyze_stock():
    symbol = request.args.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': 'Missing stock symbol'}), 400

    try:
        df = yf.download(symbol, period='3mo', interval='1d')
        if df.empty:
            return jsonify({'error': f"No data found for {symbol}"}), 404

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        close = df['Close']

        sma20 = calculate_sma(close, 20).iloc[-1]
        sma50 = calculate_sma(close, 50).iloc[-1]
        ema20 = calculate_ema(close, 20).iloc[-1]
        rsi = calculate_rsi(close).iloc[-1]
        macd, signal, histogram = calculate_macd(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)

        weekly_avg = df['Close'].resample('W').mean().dropna().round(2).tail(4).to_dict()
        weekly_avg = {str(date.date()): price for date, price in weekly_avg.items()}

        response = {
            "symbol": symbol,
            "latestPrice": round(close.iloc[-1], 2),
            "change": round(close.iloc[-1] - close.iloc[-2], 2),
            "changePercent": round((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2),
            "volume": int(df['Volume'].iloc[-1]),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "ema20": round(ema20, 2),
            "rsi": round(rsi, 2),
            "macd": round(macd.iloc[-1], 4),
            "macdSignal": round(signal.iloc[-1], 4),
            "macdHistogram": round(histogram.iloc[-1], 4),
            "bollingerBands": {
                "upper": round(bb_upper.iloc[-1], 2),
                "middle": round(bb_middle.iloc[-1], 2),
                "lower": round(bb_lower.iloc[-1], 2),
            },
            "weeklyAveragePrices": weekly_avg
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route("/api/stock", methods=["GET"])
def get_stock_data():
    symbol = request.args.get("symbol", "AAPL")
    try:
        data = yf.download(symbol, period="6mo", interval="1d")
        if data.empty:
            return jsonify({"error": "No data found"}), 404
        
        # Return latest price and indicators
        latest = data.iloc[-1]
        response = {
            "symbol": symbol,
            "latest_price": round(latest["Close"], 2),
            "sma_20": round(data["Close"].rolling(window=20).mean().iloc[-1], 2),
            "ema_20": round(data["Close"].ewm(span=20, adjust=False).mean().iloc[-1], 2)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
