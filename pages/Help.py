import streamlit as st
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد
apply_custom_css()

st.set_page_config(
    page_title="Help & Tips - Global Market Insight",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .tip-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1.5rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.1) 0%, rgba(46, 213, 115, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2ed573;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Help & Tips</h1>', unsafe_allow_html=True)

# ========================================
# GETTING STARTED
# ========================================
st.markdown("## Getting Started")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("### Technical Analysis Dashboard")
        st.markdown("**Perfect for:** Day traders, technical analysts, active investors")
        
        st.markdown("**How to use:**")
        st.markdown("""
        1. **Select Market:** Choose between ASE (Jordan) or US Stock Market.
        2. **Select Stock:**
           - *For ASE:* Choose Sector → Subsector → Company.
           - *For US:* Enter the official Ticker symbol (e.g., AAPL, TSLA, NVDA).
        3. Choose analysis period (30 to max available days).
        4. Enable technical indicators you want to see.
        5. Use the profit calculator for projections.
        6. Download data for offline analysis.
        """)
        
        st.markdown("**Best Practices:**")
        st.markdown("""
        - Use at least 180 days for reliable signals.
        - Combine RSI + MACD + Moving Averages.
        - Watch for strong signals (2/3 or 3/3 agreement).
        - Check multiple timeframes before trading.
        - Always verify with volume indicators.
        """)

with col2:
    with st.container():
        st.markdown("### AI Price Prediction")
        st.markdown("**Perfect for:** Long-term investors, data scientists, quant analysts")
        
        st.markdown("**How to use:**")
        st.markdown("""
        1. **Select Market & Stock** using the sidebar.
        2. **Choose Model Type:** Machine Learning, Deep Learning, or ARIMA.
        3. **Configure Parameters:** Adjust layers, epochs, or estimators.
        4. **Select Features:** Add technical indicators to the AI input for better accuracy.
        5. Train the model and watch the live progress.
        6. Compare different model results using the "Compare Models" dashboard.
        """)
        
        st.markdown("**Best Practices:**")
        st.markdown("""
        - **Data is King:** For US stocks, use 'max' period for deep learning.
        - Try multiple model types to find the most consistent one.
        - LSTM works best for volatile stocks (like Tech).
        - ARIMA for stable, trending stocks.
        - Validate predictions with technical analysis.
        """)

st.markdown("---")

# ========================================
# TECHNICAL INDICATORS GUIDE
# ========================================
st.markdown("## Technical Indicators Guide")

tab1, tab2, tab3, tab4 = st.tabs(["Moving Averages", "Momentum", "Volatility", "Volume"])

with tab1:
    st.markdown("""
    ### Moving Averages
    
    **Simple Moving Average (SMA)**
    - Calculates average price over specified period
    - Smooths out price fluctuations
    - Good for identifying trends
    - **Tip:** Use 20-day for short-term, 50-day for medium, 200-day for long-term
    
    **Exponential Moving Average (EMA)**
    - Gives more weight to recent prices
    - More responsive to recent changes
    - Better for fast-moving markets
    - **Tip:** 12 and 26-day EMAs are commonly used for MACD
    
    **Trading Signals:**
    - Buy when price crosses above MA
    - Sell when price crosses below MA
    - Golden Cross: 50-day MA crosses above 200-day MA (bullish)
    - Death Cross: 50-day MA crosses below 200-day MA (bearish)
    """)

with tab2:
    st.markdown("""
    ### Momentum Indicators
    
    **Relative Strength Index (RSI)**
    - Measures speed and change of price movements
    - Range: 0 to 100
    - **Overbought:** Above 70 (possible sell signal)
    - **Oversold:** Below 30 (possible buy signal)
    - **Tip:** Look for divergences between RSI and price
    
    **Moving Average Convergence Divergence (MACD)**
    - Shows relationship between two moving averages
    - MACD Line = 12 EMA - 26 EMA
    - Signal Line = 9-day EMA of MACD
    - **Buy Signal:** MACD crosses above signal line
    - **Sell Signal:** MACD crosses below signal line
    - **Tip:** Confirm with histogram (difference between MACD and signal)
    """)

with tab3:
    st.markdown("""
    ### Volatility Indicators
    
    **Bollinger Bands**
    - Consists of: Middle band (20 SMA), Upper band (+2 SD), Lower band (-2 SD)
    - Shows price volatility and potential breakouts
    - **Buy Signal:** Price touches lower band
    - **Sell Signal:** Price touches upper band
    - **Squeeze:** Bands narrow = low volatility, potential breakout coming
    - **Expansion:** Bands widen = high volatility
    
    **Average True Range (ATR)**
    - Measures market volatility
    - Higher ATR = higher volatility
    - Used for setting stop-loss levels
    - **Tip:** Set stop-loss at 2x ATR from entry
    """)

with tab4:
    st.markdown("""
    ### Volume Indicators
    
    **Volume Analysis**
    - Confirms price movements
    - High volume + price increase = strong bullish signal
    - High volume + price decrease = strong bearish signal
    - Low volume = weak signal, potential reversal
    
    **On-Balance Volume (OBV)**
    - Cumulative volume indicator
    - Rising OBV = accumulation (bullish)
    - Falling OBV = distribution (bearish)
    - **Tip:** Look for OBV divergence from price
    """)

st.markdown("---")

# ========================================
# AI MODEL SELECTION GUIDE
# ========================================
st.markdown("## AI Model Selection Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Deep Learning Models
    
    **LSTM (Long Short-Term Memory)**
    - Best for: Complex patterns, volatile stocks (US Tech)
    - Pros: Handles long-term dependencies, captures trends
    - Cons: Requires more data (200+ days), slower training
    - **When to use:** For stocks with high historical data volume.
    
    **GRU (Gated Recurrent Unit)**
    - Best for: Medium complexity, faster training
    - Pros: Similar to LSTM but faster, less parameters
    - Cons: May miss very long-term patterns
    - **When to use:** When you need quick results.
    
    **RNN (Recurrent Neural Network)**
    - Best for: Short-term predictions
    - Pros: Simple, fast training
    - Cons: Struggles with long sequences
    """)

with col2:
    st.markdown("""
    ### Machine Learning & Classical
    
    **Random Forest & XGBoost**
    - Best for: Non-linear relationships and feature importance.
    - Pros: Extremely fast, handles missing data well.
    - Cons: Can overfit if depth is too high.
    - **When to use:** When you have many technical indicators as features.
    
    **ARIMA (AutoRegressive Integrated Moving Average)**
    - Best for: Stable stocks with clear trends (Utilities, Banks)
    - Pros: Statistically robust, interpretable
    - Cons: Assumes linearity, sensitive to outliers
    
    **Linear/Ridge/Lasso Regression**
    - Best for: Simple trend continuation
    - Pros: Very fast, highly interpretable
    - Cons: Cannot capture complex market shifts
    """)

st.markdown("---")

# ========================================
# BEST PRACTICES
# ========================================
st.markdown("## Best Practices for Success")

st.markdown("### Risk Management")
st.markdown("""
- Never invest more than you can afford to lose.
- Diversify your portfolio across different sectors and **both markets** (JOD & USD).
- Set stop-loss orders for every trade using ATR indicators.
- Use position sizing (don't put all money in one stock).
""")

st.markdown("---")

# ========================================
# FAQ
# ========================================
st.markdown("## Frequently Asked Questions")

with st.expander("How do I search for a US Stock?"):
    st.markdown("""
    When you select the **US Stock Market**, a text box will appear. You must enter the official **Ticker Symbol** for the company.
    For example:
    - Apple = `AAPL`
    - Tesla = `TSLA`
    - Microsoft = `MSFT`
    - S&P 500 Index ETF = `SPY`
    """)

with st.expander("Where does the data come from?"):
    st.markdown("""
    - **ASE Data:** Pulled directly from the official Amman Stock Exchange historical archives.
    - **US Market Data:** Fetched in real-time from Yahoo Finance API, providing highly accurate split-adjusted and dividend-adjusted data.
    """)

with st.expander("How accurate are the AI predictions?"):
    st.markdown("""
    AI predictions are based on historical patterns and provide probability estimates, not certainties. 
    Accuracy varies by stock and market conditions. Always check the model's R² score and error metrics (MAPE). 
    Always combine AI predictions with the **Technical Dashboard** signals.
    """)

with st.expander("Can I download the data for my own analysis?"):
    st.markdown("""
    Yes! The Technical Dashboard allows you to download:
    - Full historical price data (Excel/CSV)
    - Technical indicator values (RSI, MACD, etc.)
    """)

st.markdown("---")

# ========================================
# TROUBLESHOOTING
# ========================================
st.markdown("## Troubleshooting")

st.markdown("""
### Common Issues and Solutions

**Issue: US Stock data not loading / "Ticker not found"**
- Verify the ticker symbol spelling. (e.g., Google is `GOOGL` or `GOOG`, not GOOGLE).
- Ensure your device has an active internet connection to reach Yahoo Finance.

**Issue: AI model training fails**
- Ensure you have at least 200 days of data.
- For ASE stocks, some newer companies might not have enough historical data for deep learning. Switch to Machine Learning (Random Forest) instead.
- Reduce the number of epochs or sequence length.

**Issue: Charts not loading**
- Refresh the page.
- Clear browser cache.

**Issue: Indicators showing conflicting signals**
- This is normal - not all indicators agree.
- Look for 2/3 or 3/3 consensus.
- Check multiple timeframes and consult the AI Chat Assistant.
""")

st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1rem;'>Need more help? Check our <b>About</b> page for detailed information</p>
    <p style='font-size: 0.9rem;'>Sources: <a href='https://www.ase.com.jo' target='_blank'>ASE Official</a> | <a href='https://finance.yahoo.com/' target='_blank'>Yahoo Finance</a></p>
</div>
""", unsafe_allow_html=True)

# Navigation
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("ASE_insight.py")