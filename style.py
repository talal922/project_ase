# style.py
import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    
        /* =========================================
           1. إعدادات السايد بار (Sidebar)
        ========================================= */
        
        /* تعديل لون خلفية السايد بار */
        [data-testid="stSidebar"] {
            background-color: #0e1117;
;
        }
        
        /* زيادة التباعد بين الأزرار داخل السايد بار */
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 15px;
        }

        /* =========================================
           2. Styling the Default Streamlit Sidebar Navigation
        ========================================= */
        
        /* Style the navigation links to look like buttons */
        [data-testid="stSidebarNav"] a {
            display: block;
            padding: 12px 16px;
            margin: 8px 0;
            background-color: #0e1117
;
            color: #222222;
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            border: 1px solid transparent;
        }
        
        /* Hover effect for navigation links */
        [data-testid="stSidebarNav"] a:hover {
            background-color: #3d9df3;
            color:  #888;
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
            border-color: #6e7848;
        }
        
        /* Active page styling */
        [data-testid="stSidebarNav"] a[data-testid="stSidebarNavLink-active"] {
            background-color: #667eea;
            color: white;
            font-weight: bold;
            border-color: #5a67d8;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.5);
        }

        /* =========================================
           3. إعدادات الأزرار العامة في كل الصفحات
        ========================================= */
        
        .stButton > button {
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            border-color: #667eea;
            color: #667eea;
        }

        /* =========================================
           4. إعدادات التبويبات (Tabs)
        ========================================= */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
        }
                 .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
    }
      
       
    </style>
    """, unsafe_allow_html=True)