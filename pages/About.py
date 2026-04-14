import streamlit as st
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد

st.set_page_config(
    page_title="About - Global Market Insight",
    page_icon="",
    layout="wide"
)

apply_custom_css()

st.markdown('<h1 class="main-title">About Global Market Insight</h1>', unsafe_allow_html=True)

# ========================================
# PLATFORM OVERVIEW
# ========================================
st.markdown("""
<div class="section-card">
    <div class="section-title">Platform Overview</div>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #ccc;'>
        <b>Global Market Insight</b> is a professional trading and analytics platform designed 
        for both the <b>Amman Stock Exchange (ASE)</b> and the <b>US Stock Market (Wall Street)</b>. 
        Built with cutting-edge technology, the platform combines advanced technical analysis with 
        artificial intelligence to provide traders and investors with powerful decision-making tools.
    </p>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #aaa;'>
        Our mission is to democratize access to professional-grade trading tools and make 
        sophisticated market analysis accessible to everyone—from local Jordanian investors 
        to global day traders aiming for Wall Street.
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# KEY FEATURES
# ========================================
st.markdown("""
<div class="section-card">
    <div class="section-title">Key Features</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Technical Analysis
    - Dual-Market Support (ASE & US Stocks)
    - Real-time candlestick charts
    - 15+ technical indicators
    - Moving averages (SMA, EMA)
    - RSI, MACD, Bollinger Bands
    - Volume & Volatility analysis
    - AI-powered trading signals
    - Historical data export (Excel/CSV)
    """)
    
    st.markdown("""
    ### Data & Analytics
    - Historical data from 30 days up to **Max Available (20+ years for US)**
    - Statistical performance metrics
    - Investment profit calculator
    - Volatility and Return analysis
    - Real-time market status and clocks
    """)

with col2:
    st.markdown("""
    ### AI Price Prediction
    - **Deep Learning:** LSTM, GRU, RNN
    - **Machine Learning:** XGBoost, Random Forest, LightGBM, SVR
    - **Classical:** AR, MA, ARMA, ARIMA
    - Custom model architecture & hyperparameter tuning
    - Forecast up to 30 days ahead
    - Comprehensive Model comparison dashboard
    - Multi-feature training (Prices + Indicators)
    """)
    
    st.markdown("""
    ### Smart AI Assistant
    - GPU-powered financial chatbot (Gemini / Llama)
    - Deep knowledge of ASE and Wall Street
    - Instant analysis of technical patterns
    - Risk management guidance
    - Sector-specific insights
    """)

st.markdown("---")

# ========================================
# TECHNICAL SPECIFICATIONS
# ========================================
st.markdown("## Technical Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Data Coverage
    - **Markets:** ASE (Jordan) & US Stock Market
    - **Companies:** 194+ ASE | 5000+ US Tickers
    - **Historical Data:** Up to 20+ years (US), 2+ years (ASE)
    - **Update Frequency:** Real-time / Daily
    - **Data Sources:** ASE Official & Yahoo Finance
    """)

with col2:
    st.markdown("""
    ### Technology Stack
    - **Frontend:** Streamlit
    - **Charts:** Plotly & Matplotlib
    - **AI/ML:** PyTorch, Scikit-Learn, XGBoost
    - **Data Processing:** Pandas, NumPy, TA-Lib
    - **LLM Engine:** Google Gemini API
    """)

with col3:
    st.markdown("""
    ### AI Architecture
    - **Neural Networks:** Bidirectional LSTM, GRU
    - **Evaluation Metrics:** MSE, MAE, RMSE, R², MAPE
    - **Validation:** Out-of-sample testing, Shift correlation
    - **Optimization:** Adam Optimizer, Early Stopping
    """)

st.markdown("---")

# ========================================
# VERSION INFORMATION
# ========================================
st.markdown("## Version Information")

version_info = {
    "Version": ["3.0", "2.0", "1.5", "1.0"],
    "Release Date": ["March 2026", "December 2025", "September 2025", "June 2025"],
    "Major Features": [
        "US Market Integration, Machine Learning Models, Global AI Chat, Advanced Model Comparison",
        "AI Prediction, Enhanced UI, Deep Learning Models",
        "Technical Dashboard, 15+ Indicators",
        "Initial Release, Basic ASE Charts"
    ],
    "Status": ["Current", "Previous", "Deprecated", "Archived"]
}

st.table(version_info)

st.markdown("---")

# ========================================
# RESOURCES & LINKS
# ========================================
st.markdown("## Resources & Links")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Official ASE Resources
    - [ASE Official Website](https://www.ase.com.jo)
    - [Trading Guide](https://www.ase.com.jo/en/page/trading-guide)
    - [Historical Data](https://www.ase.com.jo/en/historical-data)
    """)

with col2:
    st.markdown("""
    ### US Market Resources
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [S&P 500 Overview](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)
    - [Bloomberg Markets](https://www.bloomberg.com/markets)
    """)

with col3:
    st.markdown("""
    ### Learning Resources
    - [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
    - [AI in Trading](https://www.investopedia.com/artificial-intelligence-ai-in-finance-5088543)
    - [Risk Management](https://www.investopedia.com/articles/trading/09/risk-management.asp)
    """)

st.markdown("---")

# ========================================
# DISCLAIMER
# ========================================
st.markdown("""
<div class="section-card" style="border-left-color: #ff6b6b;">
    <div class="section-title" style="color: #ff6b6b;">Important Disclaimer</div>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <b>Global Market Insight</b> is provided for educational and informational purposes only. 
        The platform does not provide financial, investment, or trading advice. 
        All information, predictions, and signals are generated through automated 
        algorithms and should not be considered as recommendations to buy or sell securities.
    </p>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <b>Users should:</b>
    </p>
    <ul style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <li>Conduct their own research and due diligence</li>
        <li>Consult with licensed financial advisors</li>
        <li>Understand that past performance does not guarantee future results</li>
        <li>Be aware that all investments carry risk</li>
        <li>Never invest more than they can afford to lose</li>
    </ul>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        By using this platform, you acknowledge that you understand and accept these terms.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# CONTACT & SUPPORT
# ========================================
st.markdown("## Contact & Support")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Platform Support
    For technical issues or questions about using the platform.
    
    **Response Time:** 24-48 hours
    """)

with col2:
    st.markdown("""
    ### Feature Requests
    Have an idea for a new feature? We'd love to hear from you.
    
    **Status:** Open for suggestions
    """)

with col3:
    st.markdown("""
    ### Bug Reports
    Found a bug? Please report it so we can fix it quickly.
    
    **Priority:** High
    """)

st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1rem;'><b>Global Market Insight Platform v3.0</b></p>
    <p style='font-size: 0.9rem;'>Built with dedication for Jordanian & Global Traders</p>
    <p style='font-size: 0.8rem; color: #666; margin-top: 1rem;'>
        © 2026 Global Market Insight. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("ASE_insight.py")