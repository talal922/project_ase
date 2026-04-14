import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ase_data import sectors, company_symbols, company_export_ids
from ase_utils import (fetch_full_history, calculate_trading_signals, 
                       calculate_metrics, add_technical_indicators, fetch_us_history)
import pandas as pd
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Technical Dashboard",
    layout="wide"
)

apply_custom_css()

st.title("Technical Analysis Dashboard")

# ========== HEADER ==========
st.markdown('<p class="main-header"> Professional Trading Platform (ASE & US Markets)</p>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("##  Market Selection")
    market_choice = st.radio("Select Market", ["Amman Stock Exchange (ASE)", "US Stock Market"])
    st.markdown("---")
    
    st.markdown("##  Stock Selection")
    
    if market_choice == "Amman Stock Exchange (ASE)":
        selected_sector = st.selectbox(" Sector", list(sectors.keys()))
        selected_subsector = st.selectbox(" Subsector", list(sectors[selected_sector].keys()))
        selected_company = st.selectbox(" Company", sectors[selected_sector][selected_subsector])
        symbol = company_symbols.get(selected_company)
    else:
        # السوق الأمريكي
        selected_company = st.text_input("Enter US Ticker (e.g., AAPL, MSFT, TSLA, NVDA)", "AAPL").upper()
        symbol = selected_company
    
    st.markdown("---")
    st.markdown("##  Settings")
    period_days = st.slider(" Analysis Period (days)", 30, 5000, 180, 30)
    
    st.markdown("---")
    st.markdown("##  Indicators")
    
    with st.expander(" Trend"):
        show_sma = st.checkbox("SMA")
        show_ema = st.checkbox("EMA")
        show_bbands = st.checkbox("Bollinger Bands")
    
    with st.expander(" Momentum"):
        show_rsi = st.checkbox("RSI")
        show_macd = st.checkbox("MACD")
        show_stochastic = st.checkbox("Stochastic")
    
    with st.expander(" Volume & Volatility"):
        show_volume = st.checkbox("Volume")
        show_obv = st.checkbox("OBV")
        show_atr = st.checkbox("ATR")
        show_adx = st.checkbox("ADX")

# ========== LOAD DATA ==========
if not symbol:
    st.error(" Company symbol not found")
    st.stop()

@st.cache_data(ttl=3600)
def load_data(symbol, market):
    if market == "Amman Stock Exchange (ASE)":
        return fetch_full_history(symbol, company_export_ids)
    else:
        # جلب أقصى بيانات ممكنة للسوق الأمريكي (max) أو تحديد فترة كبيرة مثل "10y"
        return fetch_us_history(symbol, period_days="5y")

with st.spinner(f"Loading {selected_company}..."):
    df_full = load_data(symbol, market_choice)  # تمرير نوع السوق

if df_full is None or df_full.empty:
    st.error(" No data available. Please check the ticker symbol or your internet connection.")
    st.stop()

# Process data - filter for display only
df = df_full.tail(period_days).copy()
df = add_technical_indicators(df)
metrics = calculate_metrics(df)
signals = calculate_trading_signals(df)

# ========== METRICS DASHBOARD ==========
st.markdown(f"### {selected_company} ({symbol})")
st.markdown("####  Key Performance Indicators")

# تحديد العملة بناءً على السوق
currency = "JOD" if market_choice == "Amman Stock Exchange (ASE)" else "USD"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(" Price", f"{currency} {metrics['current_price']:.3f}", 
            f"{metrics['daily_change']:+.3f} ({metrics['daily_change_pct']:+.2f}%)")
col2.metric(" Return", f"{metrics['total_return']:+.2f}%", f"{period_days} days")
col3.metric(" Volatility", f"{metrics['volatility']:.2f}%", "Annualized")
col4.metric(" Sharpe", f"{metrics['sharpe_ratio']:.2f}", "Risk-Adjusted")
col5.metric("Drawdown", f"{metrics['max_drawdown']:.2f}%", "Max Risk")

col1, col2 = st.columns(2)
col1.info(f"**52W Range:** {currency} {metrics['low_52w']:.3f} - {metrics['high_52w']:.3f}")
col2.info(f"**Avg Volume:** {metrics['avg_volume']:,.0f} | **Avg Value:** {currency} {metrics['avg_value']:,.0f}")

st.markdown("---")

# ========== TRADING SIGNALS ==========
st.markdown("####  AI Trading Signals")
latest = signals.iloc[-1]
current_rsi = df["RSI"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)

with col1:
    buy = int(latest['Total_Buy'])
    if buy >= 2:
        st.markdown(f"### STRONG BUY\n**Strength:** {buy}/3")
    elif buy == 1:
        st.markdown(f"### MODERATE BUY\n**Strength:** {buy}/3")
    else:
        st.markdown("### NEUTRAL")

with col2:
    sell = int(latest['Total_Sell'])
    if sell >= 2:
        st.markdown(f"### STRONG SELL\n**Strength:** {sell}/3")
    elif sell == 1:
        st.markdown(f"### MODERATE SELL\n**Strength:** {sell}/3")
    else:
        st.markdown("### HOLD")

with col3:
    if current_rsi < 30:
        st.markdown(f"### RSI: {current_rsi:.1f}\n**Oversold** ")
    elif current_rsi > 70:
        st.markdown(f"### RSI: {current_rsi:.1f}\n**Overbought** ")
    else:
        st.markdown(f"### RSI: {current_rsi:.1f}\n**Neutral** ")

with col4:
    trend = " Bullish" if df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1] else " Bearish"
    macd_sig = "Bullish Cross" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "Bearish Cross"
    st.markdown(f"### Trend\n**MA:** {trend}\n**MACD:** {macd_sig}")

st.markdown("---")

# ========== MAIN CHART ==========
st.markdown("####  Price Chart & Technical Analysis")

# Main Price Chart
fig_price = go.Figure()

# تحديد سعر الافتتاح (إذا موجود بنستخدمه زي السوق الأمريكي، وإذا مش موجود بنستخدم الإغلاق زي بورصة عمان)
open_price = df['Opening'] if 'Opening' in df.columns else df['Closing']

# Candlestick
fig_price.add_trace(go.Candlestick(
    x=df['Date'], open=open_price, high=df['High'], 
    low=df['Low'], close=df['Closing'], name="Price",
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
))

# Bollinger Bands
if show_bbands:
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["BB_High"], name="BB Upper", 
                            line=dict(color="rgba(250,128,114,0.5)", width=1)))
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["BB_Low"], name="BB Lower", 
                            line=dict(color="rgba(144,238,144,0.5)", width=1), fill='tonexty'))
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["BB_Mid"], name="BB Middle", 
                            line=dict(color="rgba(135,206,250,0.5)", width=1, dash='dot')))

# Moving Averages
if show_sma:
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", 
                            line=dict(color="#FFA500", width=2)))
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", 
                            line=dict(color="#4169E1", width=2.5)))
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200", 
                            line=dict(color="#9370DB", width=3)))

if show_ema:
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20", 
                            line=dict(color="#FF69B4", width=2, dash='dot')))
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50", 
                            line=dict(color="#00CED1", width=2.5, dash='dot')))

# Signals
buy_sig = signals[signals['Total_Buy'] >= 2]
sell_sig = signals[signals['Total_Sell'] >= 2]

if not buy_sig.empty:
    fig_price.add_trace(go.Scatter(x=buy_sig['Date'], y=buy_sig['Price'], mode='markers',
        name='Buy Signal', marker=dict(symbol='triangle-up', size=15, color='lime',
        line=dict(color='darkgreen', width=2))))

if not sell_sig.empty:
    fig_price.add_trace(go.Scatter(x=sell_sig['Date'], y=sell_sig['Price'], mode='markers',
        name='Sell Signal', marker=dict(symbol='triangle-down', size=15, color='red',
        line=dict(color='darkred', width=2))))

fig_price.update_layout(
    height=500, 
    template="plotly_dark", 
    showlegend=True,
    hovermode='x unified',
    xaxis_rangeslider_visible=False,
    title="Price Action with Indicators",
    xaxis_title="Date",
    yaxis_title=f"Price ({currency})"
)

st.plotly_chart(fig_price, use_container_width=True)

# ========== VOLUME CHART ==========
if show_volume:
    st.markdown("####  Trading Volume")
    colors = ['#ef5350' if df['Closing'].iloc[i] < df['Closing'].iloc[i-1] else '#26a69a' 
              for i in range(1, len(df))]
    colors.insert(0, '#26a69a')
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df["Date"], y=df["No. of Shares"], 
                             marker_color=colors, opacity=0.7, name="Volume"))
    fig_vol.update_layout(
        height=250,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Shares"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ========== RSI INDICATOR ==========
if show_rsi:
    st.markdown("####  RSI - Relative Strength Index")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", 
                            line=dict(color="#9370DB", width=3)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7,
                     annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7,
                     annotation_text="Oversold (30)")
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)
    fig_rsi.update_layout(
        height=250,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="RSI Value",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ========== MACD INDICATOR ==========
if show_macd:
    st.markdown("####  MACD - Moving Average Convergence Divergence")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD Line", 
                            line=dict(color="#4169E1", width=2)))
    fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal Line", 
                            line=dict(color="#FFA500", width=2)))
    
    colors = ['#ef5350' if v < 0 else '#26a69a' for v in df["MACD_Hist"]]
    fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram",
                        marker_color=colors, opacity=0.5))
    fig_macd.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
    fig_macd.update_layout(
        height=250,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="MACD Value"
    )
    st.plotly_chart(fig_macd, use_container_width=True)

st.markdown("---")
st.subheader("More Analysis & Data")
    
# ========== TABS ==========
tab1, tab2, tab3, tab4 = st.tabs([" Profit Calculator", " Statistics", " Advanced", " Data"])

with tab1:
    st.markdown("###  Investment Profit Calculator")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        amount = st.number_input(f"Investment ({currency})", 100, 1000000, 10000, 500)
    with col2:
        entry_idx = st.slider("Entry (days ago)", 0, len(df)-1, len(df)//2)
    
    entry_price = df.iloc[entry_idx]['Closing']
    entry_date = df.iloc[entry_idx]['Date']
    current_price = df.iloc[-1]['Closing']
    shares = amount / entry_price
    value = shares * current_price
    profit = value - amount
    profit_pct = (profit / amount) * 100
    days = (df.iloc[-1]['Date'] - entry_date).days
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entry", f"{currency} {entry_price:.3f}")
    col2.metric("Shares", f"{shares:.2f}")
    col3.metric("Value", f"{currency} {value:.2f}")
    col4.metric("Profit", f"{currency} {profit:+.2f}", f"{profit_pct:+.2f}%")
    
    if profit > 0:
        st.success(f" Profit of {currency} {profit:.2f} ({profit_pct:.2f}%) in {days} days")
    else:
        st.error(f" Loss of {currency} {abs(profit):.2f} ({profit_pct:.2f}%) in {days} days")
    
    period_df = df.iloc[entry_idx:].copy()
    period_df['Portfolio'] = (period_df['Closing'] / entry_price) * amount
    
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=period_df['Date'], y=period_df['Portfolio'], 
        fill='tozeroy', line=dict(color='#26a69a' if profit > 0 else '#ef5350', width=3)))
    fig_p.add_hline(y=amount, line_dash="dash", line_color="white")
    fig_p.update_layout(title="Portfolio Value", template="plotly_dark", height=350)
    st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    st.markdown("###  Statistical Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=df["Returns"].dropna()*100, nbinsx=50, marker_color='#667eea'))
        fig_d.update_layout(title="Returns Distribution", template="plotly_dark", height=300)
        st.plotly_chart(fig_d, use_container_width=True)
        
        st.dataframe({
            'Metric': ['Mean', 'Std Dev', 'Best Day', 'Worst Day', 'Win Rate'],
            'Value': [
                f"{df['Returns'].mean()*100:.4f}%",
                f"{df['Returns'].std()*100:.4f}%",
                f"{df['Returns'].max()*100:.2f}%",
                f"{df['Returns'].min()*100:.2f}%",
                f"{(df['Returns'] > 0).sum()/len(df)*100:.1f}%"
            ]
        }, hide_index=True)
    
    with col2:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=df["Date"], y=(df["Cumulative_Returns"]-1)*100, 
            fill='tozeroy', line=dict(color='#764ba2', width=2)))
        fig_c.update_layout(title="Cumulative Returns", template="plotly_dark", height=300)
        st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    st.markdown("###  Advanced Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        if show_stochastic:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K"))
            fig_s.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D"))
            fig_s.add_hline(y=80, line_dash="dash", line_color="red")
            fig_s.add_hline(y=20, line_dash="dash", line_color="green")
            fig_s.update_layout(title="Stochastic", template="plotly_dark", height=300)
            st.plotly_chart(fig_s, use_container_width=True)
        
        if show_atr:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df["Date"], y=df["ATR"], fill='tozeroy'))
            fig_a.update_layout(title="ATR (Volatility)", template="plotly_dark", height=300)
            st.plotly_chart(fig_a, use_container_width=True)
    
    with col2:
        if show_adx:
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=df["Date"], y=df["ADX"], fill='tozeroy'))
            fig_adx.add_hline(y=25, line_dash="dash", line_color="orange")
            fig_adx.update_layout(title="ADX (Trend Strength)", template="plotly_dark", height=300)
            st.plotly_chart(fig_adx, use_container_width=True)
        
        if show_obv:
            fig_o = go.Figure()
            fig_o.add_trace(go.Scatter(x=df["Date"], y=df["OBV"], fill='tozeroy'))
            fig_o.update_layout(title="On-Balance Volume", template="plotly_dark", height=300)
            st.plotly_chart(fig_o, use_container_width=True)

with tab4:
    st.markdown("###  Recent Trading Data")
    recent = df[['Date', 'Closing', 'High', 'Low', 'No. of Shares', 'No. of Trans', 'Value Traded']].tail(20).copy()
    recent['Change %'] = df['Closing'].pct_change().tail(20) * 100
    recent = recent.iloc[::-1]
    
    st.dataframe(recent.style.format({
        'Closing': '{:.3f}', 'High': '{:.3f}', 'Low': '{:.3f}',
        'Value Traded': '{:,.0f}', 'No. of Shares': '{:,.0f}',
        'No. of Trans': '{:,.0f}', 'Change %': '{:+.2f}%'
    }), use_container_width=True, height=400)

# Download
st.divider()
st.markdown("### Download Options")
col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        f" Download Filtered Data (CSV) - {len(df)} rows", 
        csv, 
        f"{selected_company}_{symbol}_filtered_{period_days}days.csv", 
        "text/csv",
        use_container_width=True,
        help=f"Download the current {period_days} days of data with indicators"
    )

with col2:
    # Add indicators to full dataset for Excel download
    df_full_with_indicators = add_technical_indicators(df_full.copy())
    
    # Create Excel file in memory
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_full_with_indicators.to_excel(writer, index=False, sheet_name='Full Trading Data')
    excel_data = output.getvalue()
    
    st.download_button(
        f" Download FULL Data (Excel) - {len(df_full)} rows",
        excel_data,
        f"{selected_company}_{symbol}_FULL_DATA.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help=f"Download ALL {len(df_full)} rows with technical indicators"
    )