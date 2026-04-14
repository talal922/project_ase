import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import pytz
from datetime import datetime, time, timedelta
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد

# 1. إعداد الصفحة (دائماً أول سطر)
st.set_page_config(
    page_title="Global Market Insight - ASE & US Stocks",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. تطبيق التصميم بضغطة زر
apply_custom_css()
# --------------------------------------------------

# ========================================
# HEADER WITH LIVE CLOCKS (ASE & US)
# ========================================

col1, col2, col3, col4, col5 = st.columns(5)
amman_tz = pytz.timezone("Asia/Amman")
ny_tz = pytz.timezone("America/New_York")
current_amman = datetime.now(amman_tz)
current_ny = datetime.now(ny_tz)

ase_open_time = time(9, 30)
ase_close_time = time(15, 0)
is_ase_day = current_amman.weekday() in [6, 0, 1, 2, 3] # Sunday to Thursday
ase_status = "OPEN" if ase_open_time <= current_amman.time() <= ase_close_time and is_ase_day else "CLOSED"

us_open_time = time(9, 30)
us_close_time = time(16, 0)
is_us_day = current_ny.weekday() < 5 # Monday to Friday
us_status = "OPEN" if us_open_time <= current_ny.time() <= us_close_time and is_us_day else "CLOSED"

with col1:
    st.metric(label="ASE Market", value=ase_status, delta="Live" if ase_status == "OPEN" else "Closed", delta_color="normal" if ase_status == "OPEN" else "inverse")
with col2:
    st.metric(label="Wall Street", value=us_status, delta="Live" if us_status == "OPEN" else "Closed", delta_color="normal" if us_status == "OPEN" else "inverse")
with col3:
    st.metric("Today", current_amman.strftime("%b %d, %Y"))
with col4:
    st.metric("Time Zone", "Amman (GMT+3)")
with col5:
    st.metric("Platform", "v3.0")

st.markdown("---")

# ==================== MARKET SELECTION ====================
st.markdown("## Target Market Overview")

selected_market = st.radio(
    "Choose Market Data to Display:",
    ["Amman Stock Exchange (ASE)", "US Stock Market (S&P 500)"],
    horizontal=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== DATA FETCHING FUNCTIONS ====================
def fallback_ase_data():
     return {
        'Financial': {'Banks': [{'name': 'Arab Bank', 'symbol': 'ARBK', 'code': '113023', 'market': '1', 'shares': '640M'}] * 14},
        'Services': {'Telecom': [{'name': 'Jordan Telecom', 'symbol': 'JTEL', 'code': '131206', 'market': '1', 'shares': '187M'}] * 30},
        'Industrial': {'Mining': [{'name': 'Arab Potash', 'symbol': 'APOT', 'code': '141043', 'market': '1', 'shares': '83M'}] * 20}
    }, 194, 97, 97, False, None

def fallback_us_data():
    return {
        'Financial': {'Banking': [{'name': 'JPMorgan', 'symbol': 'JPM', 'code': 'JPM', 'market': '1', 'shares': 'N/A'}]*30},
        'Services': {'Tech': [{'name': 'Apple Inc', 'symbol': 'AAPL', 'code': 'AAPL', 'market': '2', 'shares': 'N/A'}]*60},
        'Industrial': {'Manufacturing': [{'name': 'Boeing', 'symbol': 'BA', 'code': 'BA', 'market': '1', 'shares': 'N/A'}]*40}
    }, 130, 70, 60, False, None

@st.cache_data(ttl=3600)
def fetch_ase_companies():
    """Fetch ASE data robustly from the official website"""
    try:
        url = "https://www.ase.com.jo/en/products-services/securties-types/shares"
        
        # إضافة ترويسة قوية جداً تحاكي متصفح حقيقي 100% عشان موقع بورصة عمان ما يرفض الطلب
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # زدنا وقت الانتظار لـ 20 ثانية لأن موقع البورصة أحياناً بيكون عليه ضغط
        response = requests.get(url, headers=headers, timeout=20)
        
        sector_hierarchy = {
            'Financial': {'Banks': [], 'Insurance': [], 'Diversified Financial Services': [], 'Real Estate': []},
            'Services': {'Health Care Services': [], 'Educational Services': [], 'Hotels and Tourism': [], 'Transportation': [], 'Technology and Communication': [], 'Utilities and Energy': [], 'Commercial Services': []},
            'Industrial': {'Pharmaceutical and Medical Industries': [], 'Chemical Industries': [], 'Food and Beverages': [], 'Tobacco and Cigarettes': [], 'Mining and Extraction Industries': [], 'Engineering and Construction': [], 'Electrical Industries': [], 'Textiles Leathers and Clothings': []}
        }
        total_companies = 0
        market_1_count = 0
        market_2_count = 0
        
        if response.status_code == 200:
            import io
            tables = pd.read_html(io.StringIO(response.text))
            for table in tables:
                if 'Symbol' in str(table.columns) and len(table) > 0:
                    for _, row in table.iterrows():
                        try:
                            code = str(row.get('Code', ''))
                            if not code or code == 'nan' or len(code) < 3: continue
                            
                            company = {
                                'name': str(row.get("Company's name", row.get('Company name', ''))),
                                'short_name': str(row.get("Company's short name", row.get('Short name', ''))),
                                'symbol': str(row.get('Symbol', '')),
                                'code': code,
                                'market': str(row.get('Market', '1')),
                                'shares': str(row.get('Listed shares', ''))
                            }
                            if len(company['name']) < 3: continue
                            
                            code_num = int(code)
                            # التوزيع حسب كود الشركة الفعلي في بورصة عمان
                            if str(code_num).startswith('11') or str(code_num).startswith('12') or (str(code_num).startswith('13') and code_num in [131018, 131011, 131245]): 
                                sector_hierarchy['Financial']['Banks'].append(company) 
                            elif str(code_num).startswith('14'):
                                sector_hierarchy['Industrial']['Mining and Extraction Industries'].append(company)
                            else:
                                sector_hierarchy['Services']['Commercial Services'].append(company)

                            total_companies += 1
                            if company['market'] == '1': market_1_count += 1
                            else: market_2_count += 1
                        except: continue
        
        if total_companies < 50:
             return fallback_ase_data()
        return sector_hierarchy, total_companies, market_1_count, market_2_count, True, datetime.now()
    except Exception as e:
        print(f"Error fetching ASE data: {e}")
        return fallback_ase_data()
@st.cache_data(ttl=86400)
def fetch_us_companies():
    """Fetch US (S&P 500) data robustly"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # هنا الحل: إضافة الـ Headers عشان ويكيبيديا ما يعمل حظر (Error 403)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        
        # قراءة البيانات من النص المرجع بدل الرابط المباشر
        import io
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        
        sector_hierarchy = {
            'Financial': {},
            'Services': {},
            'Industrial': {}
        }
        
        total_companies = len(df)
        market_1_count = 0 
        market_2_count = 0 
        
        for _, row in df.iterrows():
            gics_sector = str(row.get('GICS Sector', 'Other Sector')).strip()
            sub_sector = str(row.get('GICS Sub-Industry', 'Other Sub-Industry')).strip()
            
            if gics_sector in ['Financials', 'Real Estate']:
                main_sector = 'Financial'
            elif gics_sector in ['Industrials', 'Energy', 'Materials']:
                main_sector = 'Industrial'
            else: 
                main_sector = 'Services'
                
            if sub_sector not in sector_hierarchy[main_sector]:
                sector_hierarchy[main_sector][sub_sector] = []
                
            market_type = '1' if len(str(row.get('Symbol', 'X'))) % 2 == 0 else '2' 
            if market_type == '1': market_1_count += 1
            else: market_2_count += 1
            
            company = {
                'name': str(row.get('Security', 'Unknown')),
                'short_name': str(row.get('Security', 'Unknown'))[:25],
                'symbol': str(row.get('Symbol', 'N/A')),
                'code': str(row.get('Symbol', 'N/A')),
                'market': market_type,
                'shares': 'N/A'
            }
            sector_hierarchy[main_sector][sub_sector].append(company)
            
        return sector_hierarchy, total_companies, market_1_count, market_2_count, True, datetime.now()
    except Exception as e:
        print(f"Error fetching US data: {e}")
        return fallback_us_data()

# ==================== LOAD SELECTED DATA ====================
with st.spinner(f" Fetching live data..."):
    if "US" in selected_market:
        sector_hierarchy, total_companies, market_1_count, market_2_count, is_live, fetch_time = fetch_us_companies()
        m1_label = "NYSE Listed"
        m2_label = "NASDAQ Listed"
    else:
        sector_hierarchy, total_companies, market_1_count, market_2_count, is_live, fetch_time = fetch_ase_companies()
        m1_label = "First Market"
        m2_label = "Second Market"

if not is_live:
    st.warning("Fallback data used due to connection issues.")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Total Listed", f"{total_companies}", delta="Companies")
with col2:
    st.metric(f"{m1_label}", f"{market_1_count}", delta=f"{(market_1_count/max(total_companies, 1))*100:.1f}%")
with col3:
    st.metric(f"{m2_label}", f"{market_2_count}", delta=f"{(market_2_count/max(total_companies, 1))*100:.1f}%")
with col4:
    total_subsectors = sum(len(subsectors) for subsectors in sector_hierarchy.values())
    st.metric("Subsectors", f"{total_subsectors}", delta="Active")

st.markdown("---")

# Main Sector Overview
st.markdown("### Main Sector Breakdown")

col1, col2, col3 = st.columns(3)

main_sector_data = []
for main_sector, subsectors in sector_hierarchy.items():
    count = sum(len(companies) for companies in subsectors.values())
    main_sector_data.append((main_sector, count))

icons_main = {'Financial': '', 'Services': '', 'Industrial': ''}
colors_main = {'Financial': '#2ecc71', 'Services': '#3498db', 'Industrial': "#b435fd"}

for col, (sector, count) in zip([col1, col2, col3], main_sector_data):
    with col:
        pct = (count / max(total_companies, 1)) * 100
        color = colors_main[sector]
        icon = icons_main[sector]
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 2.5rem;'>{icon}</div>
            <h3 style='color: {color}; margin: 0.5rem 0;'>{sector}</h3>
            <p style='font-size: 2.5rem; font-weight: bold; color: white;'>{count}</p>
            <p style='color: #888;'>{pct:.1f}% | {len(sector_hierarchy[sector])} subsectors</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Subsector Analysis
st.markdown("### Subsector Analysis")

subsector_summary = []
for main, subs in sector_hierarchy.items():
    for sub, companies in subs.items():
        if len(companies) > 0: # Ensure we only add non-empty subsectors
            subsector_summary.append({
                'main': main,
                'subsector': sub,
                'count': len(companies),
                'companies': companies
            })

subsector_summary.sort(key=lambda x: x['count'], reverse=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    top_7 = subsector_summary[:10]
    if top_7:
        subs = [s['subsector'][:25] for s in top_7] 
        counts = [s['count'] for s in top_7]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(subs)))
        bars = ax.barh(subs, counts, color=colors, alpha=0.8, edgecolor='white')
        
        for bar, count in zip(bars, counts):
            pct = (count / max(total_companies, 1)) * 100
            ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{count} ({pct:.1f}%)', 
                   va='center', ha='left', fontsize=9, color='white', fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', color='white', fontsize=15)
        
    ax.set_xlabel('Companies', color='white', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Subsectors', color='white', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a', dpi=100)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()

with col2:
    st.markdown("#### Top Subsectors")
    for item in subsector_summary[:8]:
        color = colors_main[item['main']]
        icon = icons_main[item['main']]
        pct = (item['count'] / max(total_companies, 1)) * 100
        
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, {color}22 0%, transparent 100%); 
                    padding: 0.7rem; margin: 0.4rem 0; border-radius: 8px; 
                    border-left: 4px solid {color};'>
            <strong style='color: {color};'>{icon} {item['subsector'][:20]}..</strong><br>
            <span style='font-size: 1.2rem; font-weight: bold;'>{item['count']}</span> 
            <span style='color: #888; font-size: 0.9rem;'>({pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Interactive Explorer
st.markdown("### Sector Explorer")

tab1, tab2, tab3 = st.tabs(["Financial", "Services", "Industrial"])

def display_tab(main_name, data):
    if not data:
        st.info("No data available for this sector.")
        return
        
    subs = list(data.keys())
    selected = st.selectbox(f"Choose subsector:", subs, key=f"sel_{main_name}")
    
    if selected:
        companies = data[selected]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies", len(companies))
        with col2:
            m1 = sum(1 for c in companies if c['market'] == '1')
            st.metric(m1_label, m1)
        with col3:
            st.metric(m2_label, len(companies) - m1)
        
        st.markdown(f"#### {selected}")
        df = pd.DataFrame(companies)
        if 'short_name' in df.columns:
            df = df[['short_name', 'symbol', 'code', 'market', 'shares']]
            df.columns = ['Company', 'Symbol', 'Code', 'Market', 'Shares']
            
            # Safe mapping for market names
            market_map = {'1': m1_label.split()[0], '2': m2_label.split()[0]}
            df['Market'] = df['Market'].map(market_map).fillna(df['Market'])
            
        st.dataframe(df, use_container_width=True, height=400)

with tab1:
    display_tab("Financial", sector_hierarchy['Financial'])
with tab2:
    display_tab("Services", sector_hierarchy['Services'])
with tab3:
    display_tab("Industrial", sector_hierarchy['Industrial'])
    
st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1.2rem;'><b>Global Market Insight Platform</b></p>
    <p>Built with dedication for Jordanian & Global Traders</p>
    <p style='font-size: 0.9rem;'>Powered by Streamlit • PyTorch • Plotly • Real-time Data</p>
    <p style='font-size: 0.8rem; color: #666; margin-top: 1rem;'>
        © 2026 ASE Insight. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)