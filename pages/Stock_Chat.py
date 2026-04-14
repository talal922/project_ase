import streamlit as st
from google import genai
from datetime import datetime
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد
apply_custom_css()
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Global Market AI Assistant",
    page_icon="",
    layout="centered"
)

# --------------------------------------------------
# قراءة المفتاح بشكل آمن من ملف secrets.toml
# --------------------------------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("لم يتم العثور على مفتاح API. تأكد من إنشاء ملف .streamlit/secrets.toml ووضع المفتاح بداخله.")
    st.stop()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #008000 0%, #00000 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<h1 class="main-title">Global Market AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Expert Advisor for ASE & Wall Street</p>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# SYSTEM PROMPT & CONTEXT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a top-tier, professional financial market analyst and AI advisor.
You specialize in:
- Amman Stock Exchange (ASE)
- US Stock Market (Wall Street, NYSE, NASDAQ, S&P 500)
- Technical analysis (Indicators, Chart Patterns)
- Fundamental analysis
- Macroeconomics
- Risk management

Behavior rules:
1. If the user's question is related to the Amman Stock Exchange (ASE), answer with detailed ASE-specific analysis.
2. If the question is related to the US Market (e.g., Apple, Tesla, S&P 500), provide deep Wall Street insights.
3. NEVER refuse a question because it is not about a specific market. You handle BOTH local (Jordan) and global markets.
4. Use clear, professional, educational language. Format your responses with bullet points and bold text for readability.
5. Do NOT provide guaranteed profits.
6. Always mention risks when discussing investments.
"""

MARKET_CONTEXT = """
Market Knowledge Base:

1. Amman Stock Exchange (ASE):
- Country: Jordan
- Market Type: Emerging Market
- Currency: Jordanian Dinar (JOD)
- Main Index: ASE General Index
- Trading Days: Sunday to Thursday
- Trading Hours: 10:00 AM – 12:00 PM local time
- Characteristics: Low to medium liquidity, price movements can be sharp.
- Main Sectors: Banking, Insurance, Mining & Extraction, Industrial, Services.

2. US Stock Market (Wall Street):
- Country: USA
- Currency: US Dollar (USD)
- Main Indexes: S&P 500, NASDAQ Composite, Dow Jones Industrial Average
- Trading Days: Monday to Friday
- Trading Hours: 9:30 AM – 4:00 PM Eastern Time (ET)
- Characteristics: Highly liquid, global economic impact, technology-heavy (NASDAQ).
"""

ANALYSIS_STRUCTURE = """
When giving market or investment analysis, use this structure when appropriate:
- Market / Asset Overview:
- Current Trend:
- Key Supporting Factors:
- Risks & Weaknesses:
- Technical Perspective (if applicable):
- Conclusion:
"""

# --------------------------------------------------
# LOAD GEMINI CLIENT
# --------------------------------------------------
@st.cache_resource
def load_gemini():
    return genai.Client(api_key=GEMINI_API_KEY)

client = load_gemini()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if user_prompt := st.chat_input("Ask about ASE stocks, US Tech giants (AAPL, TSLA), indicators, or strategies..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing global markets..."):
            final_prompt = f"{SYSTEM_PROMPT}\n{MARKET_CONTEXT}\n{ANALYSIS_STRUCTURE}\nUser Question:\n{user_prompt}"
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=final_prompt
                )
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")

# ========================================
# SIDEBAR - NAVIGATION WITH CHAT & QUICK STATS
# ========================================
with st.sidebar:
    
    # أزرار الأسئلة السريعة
    st.markdown("### Quick Questions")
    quick_questions = {
        "ASE Outlook": "Assess the current outlook of the Amman Stock Exchange and its major sectors.",
        "US Tech Sector": "Analyze the current trends in the US Technology sector (NASDAQ).",
        "Technical Indicators": "Explain how to effectively use RSI and MACD together for trading.",
        "Risk Management": "What are the best risk management practices when trading highly volatile stocks?"
    }

    for label, question in quick_questions.items():
        if st.button(label, key=f"btn_{label}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=f"{SYSTEM_PROMPT}\n{MARKET_CONTEXT}\n{ANALYSIS_STRUCTURE}\n{question}"
                )
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
            except Exception as e:
                st.error(f"Error connecting to AI: {e}")

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.success("AI Status: **Online & Ready**")
    st.markdown(f"Date: **{datetime.now().strftime('%b %d, %Y')}**")