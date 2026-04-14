

import pandas as pd
import requests
from io import BytesIO
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import yfinance as yf

def fetch_full_history(symbol, company_export_ids):
    """Fetch historical data from ASE"""
    if not symbol or symbol not in company_export_ids:
        return pd.DataFrame()
    
    export_id = company_export_ids[symbol]
    url = f"https://ase.com.jo/en/daily-historical-export/{export_id}?destination=/en/company_historical/{symbol}&_format=xlsx"
    
    try:
        resp = requests.get(url, timeout=30)
        df = pd.read_excel(BytesIO(resp.content))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data: {e}")

def calculate_trading_signals(df):
    """Calculate buy/sell signals based on multiple indicators"""
    signals = pd.DataFrame(index=df.index)
    signals['Date'] = df['Date']
    signals['Price'] = df['Closing']
    
    # RSI signals
    rsi = RSIIndicator(df["Closing"]).rsi()
    signals['RSI_Buy'] = (rsi < 30).astype(int)
    signals['RSI_Sell'] = (rsi > 70).astype(int)
    
    # MACD signals
    macd = MACD(df["Closing"])
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    signals['MACD_Buy'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    signals['MACD_Sell'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    
    # Moving Average Crossover
    sma_short = SMAIndicator(df["Closing"], window=20).sma_indicator()
    sma_long = SMAIndicator(df["Closing"], window=50).sma_indicator()
    signals['MA_Buy'] = ((sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))).astype(int)
    signals['MA_Sell'] = ((sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))).astype(int)
    
    # Composite signal
    signals['Total_Buy'] = signals['RSI_Buy'] + signals['MACD_Buy'] + signals['MA_Buy']
    signals['Total_Sell'] = signals['RSI_Sell'] + signals['MACD_Sell'] + signals['MA_Sell']
    
    return signals

def calculate_metrics(df):
    """Calculate all performance metrics"""
    if len(df) < 2:
        return {}
    
    current_price = df['Closing'].iloc[-1]
    prev_price = df['Closing'].iloc[-2]
    start_price = df['Closing'].iloc[0]
    
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100
    total_return = ((current_price - start_price) / start_price) * 100
    
    # Returns & Volatility
    returns = df['Closing'].pct_change()
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio
    avg_return = returns.mean() * 252
    sharpe = (avg_return - 0.05) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Max Drawdown
    rolling_max = df['Closing'].expanding().max()
    drawdown = (df['Closing'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # 52-week high/low
    period_52w = min(252, len(df))
    high_52w = df['Closing'].tail(period_52w).max()
    low_52w = df['Closing'].tail(period_52w).min()
    
    return {
        'current_price': current_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'avg_volume': df['No. of Shares'].mean(),
        'avg_value': df['Value Traded'].mean()
    }

def add_technical_indicators(df):
    """Add all technical indicators to dataframe"""
    # Momentum
    df["RSI"] = RSIIndicator(df["Closing"]).rsi()
    
    # Trend
    macd = MACD(df["Closing"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    df["SMA20"] = SMAIndicator(df["Closing"], window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(df["Closing"], window=50).sma_indicator()
    df["SMA200"] = SMAIndicator(df["Closing"], window=200).sma_indicator()
    df["EMA20"] = EMAIndicator(df["Closing"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["Closing"], window=50).ema_indicator()
    
    # Volatility
    bb = BollingerBands(df["Closing"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Low"] = bb.bollinger_lband()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Closing"]).average_true_range()
    
    # Volume
    df["OBV"] = OnBalanceVolumeIndicator(df["Closing"], df["No. of Shares"]).on_balance_volume()
    
    # Advanced
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Closing"]).adx()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Closing"])
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    # Returns
    df["Returns"] = df["Closing"].pct_change()
    df["Cumulative_Returns"] = (1 + df["Returns"]).cumprod()
    
    return df
def fetch_us_history(ticker, period_days="2y"):
    """
    Fetch historical data for US stocks using yfinance
    and normalize columns to match the ASE format.
    """
    try:
        stock = yf.Ticker(ticker)
        # جلب البيانات
        df = stock.history(period=period_days)
        
        if df.empty:
            return pd.DataFrame()
            
        # تحويل الـ Index (التاريخ) إلى عمود عادي
        df = df.reset_index()
        
        # توحيد أسماء الأعمدة عشان تتطابق مع كود بورصة عمان وتشتغل المؤشرات بدون مشاكل
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Opening",
            "High": "High",
            "Low": "Low",
            "Close": "Closing",
            "Volume": "No. of Shares"
        })
        
        # إزالة الـ Timezone من التاريخ عشان ما يعمل مشاكل مع رسوم Plotly
        if 'Date' in df.columns and pd.api.types.is_datetime64tz_dtype(df['Date']):
            df['Date'] = df['Date'].dt.tz_localize(None)
            
        # إضافة أعمدة افتراضية عشان ما يضرب الكود اللي بيعتمد على بورصة عمان
        if 'No. of Trans' not in df.columns:
            df['No. of Trans'] = 0 
        if 'Value Traded' not in df.columns:
            df['Value Traded'] = df['Closing'] * df['No. of Shares']
            
        return df
        
    except Exception as e:
        print(f"Error fetching US market data for {ticker}: {e}")
        return pd.DataFrame()