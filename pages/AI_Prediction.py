import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import torch
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc
import requests
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ase_data import *
# أضفنا دالة fetch_us_history هنا 👈
from ase_utils import fetch_full_history, calculate_trading_signals, calculate_metrics, fetch_us_history
from material2 import (
    LSTMModel, GRUModel, RNNModel, TransformerModel,
    TimeSeriesTrainer, ARIMATrainer, MLTrainer,set_seed,
    DEVICE
)
warnings.filterwarnings('ignore')
from style import apply_custom_css  # 👈 استدعاء الدالة من ملفك الموحد
apply_custom_css()

# Page configuration
st.set_page_config(page_title="AI Stock Predictor", page_icon="", layout="wide")

# Session state initialization
if 'model_results' not in st.session_state:
    st.session_state.model_results = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False

# ==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=3600)
def load_data(symbol, market):
    """Load data based on the selected market"""
    if market == "Amman Stock Exchange (ASE)":
        return fetch_full_history(symbol, company_export_ids)
    else:
        # جلب أقصى بيانات ممكنة للسوق الأمريكي لتدريب الذكاء الاصطناعي بشكل ممتاز
        return fetch_us_history(symbol, period_days="5y")

def validate_data(df, date_column, price_column):
    """Validate data quality"""
    issues = []
    
    # Check for missing values
    missing_dates = df[date_column].isnull().sum()
    missing_prices = df[price_column].isnull().sum()
    
    if missing_dates > 0:
        issues.append(f"{missing_dates} missing values in date column")
    if missing_prices > 0:
        issues.append(f"{missing_prices} missing values in price column")
    
    # Check for negative prices
    if df[price_column].min() <= 0:
        issues.append(" Price column contains negative or zero values")
    
    # Check for outliers
    q1 = df[price_column].quantile(0.25)
    q3 = df[price_column].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[price_column] < (q1 - 3 * iqr)) | (df[price_column] > (q3 + 3 * iqr))).sum()
    if outliers > 0:
        issues.append(f"{outliers} potential outliers detected")
    
    return issues

def parse_dates(df, date_column):
    """Smart date parsing with multiple format attempts"""
    try:
        # Try automatic format detection first
        df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)
    except:
        try:
            # Try common formats
            df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d')
        except:
            try:
                df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
            except Exception as e:
                st.error(f" Could not parse dates: {str(e)}")
                return None
    return df

def calculate_technical_indicators(df, price_col):
    """Calculate technical indicators"""
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    rolling_mean = df[price_col].rolling(window=20).mean()
    rolling_std = df[price_col].rolling(window=20).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    return df

# ==================== UI ====================
st.title("Advanced Stock Price Prediction")
st.markdown("Train AI models for stock price forecasting: Machine Learning and deep learning (LSTM/GRU/RNN) or Classical Time Series (ARIMA)")

# ==================== MODEL COMPARISON DISPLAY ====================
if st.session_state.show_comparison and len(st.session_state.model_results) > 0:
    st.header(" Model Comparison Dashboard")
    
# Create comparison dataframe
    comparison_data = []
    for result in st.session_state.model_results:
        # حساب دقة الاتجاه (Directional Accuracy)
        y_t = np.array(result['metrics']['y_true']).flatten()
        y_p = np.array(result['metrics']['y_pred']).flatten()
        
        if len(y_t) > 1:
            # حركة ذكية: أسعار الأسهم مستحيل تكون سالبة
            # فإذا كان في أرقام سالبة، معناها إحنا بنتعامل مع (Returns)
            if np.min(y_t) < 0: 
                # حساب الاتجاه لنسبة التغير (Returns)
                dir_acc = np.mean(np.sign(y_t) == np.sign(y_p)) * 100
            else:
                # حساب الاتجاه للسعر المطلق (Closing)
                dir_acc = np.mean(np.sign(np.diff(y_t)) == np.sign(np.diff(y_p))) * 100
        else:
            dir_acc = 0.0

        comparison_data.append({
            'Model': result['model_name'],
            'RMSE': result['metrics']['rmse'],
            'MAE': result['metrics']['mae'],
            'R²': result['metrics']['r2'],
            'MAPE (%)': result['metrics']['mape'],
            'Dir Acc (%)': dir_acc,
            'Trained At': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Category': result['config']['model_category']
        })
    comparison_df = pd.DataFrame(comparison_data)
    
    # Best models summary at the top
    st.subheader(" Best Performing Models")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_rmse_idx = comparison_df['RMSE'].idxmin()
        st.metric("Best RMSE", 
                  comparison_df.loc[best_rmse_idx, 'Model'],
                  f"${comparison_df.loc[best_rmse_idx, 'RMSE']:.4f}",
                  delta_color="off")
    
    with col2:
        best_mae_idx = comparison_df['MAE'].idxmin()
        st.metric("Best MAE", 
                  comparison_df.loc[best_mae_idx, 'Model'],
                  f"${comparison_df.loc[best_mae_idx, 'MAE']:.4f}",
                  delta_color="off")
    
    with col3:
        best_r2_idx = comparison_df['R²'].idxmax()
        st.metric("Best R²", 
                  comparison_df.loc[best_r2_idx, 'Model'],
                  f"{comparison_df.loc[best_r2_idx, 'R²']:.4f}",
                  delta_color="off")
    
    with col4:
        best_mape_idx = comparison_df['MAPE (%)'].idxmin()
        st.metric("Best MAPE", 
                  comparison_df.loc[best_mape_idx, 'Model'],
                  f"{comparison_df.loc[best_mape_idx, 'MAPE (%)']:.2f}%",
                  delta_color="off")
    
    st.markdown("---")
    
    # Display metrics table with styling
    st.subheader(" Detailed Performance Metrics")
    st.dataframe(comparison_df.style.highlight_min(
        subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen'
    ).highlight_max(
        subset=['R²', 'Dir Acc (%)'], color='lightgreen' # 👈 أضفنا تمييز لأعلى دقة اتجاه
    ).format({
        'RMSE': '${:.4f}',
        'MAE': '${:.4f}',
        'R²': '{:.4f}',
        'MAPE (%)': '{:.2f}%',
        'Dir Acc (%)': '{:.2f}%' # 👈 تنسيق مئوي
    }), use_container_width=True)
    
    st.markdown("---")
    
    # Main comparison charts - 2x2 grid
    st.subheader(" Performance Comparison Charts")
    
    fig_main = plt.figure(figsize=(16, 8))
    gs = fig_main.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. RMSE Comparison
    ax1 = fig_main.add_subplot(gs[0, 0])
    colors_rmse = ['#2ecc71' if i == best_rmse_idx else '#3498db' for i in range(len(comparison_df))]
    bars1 = ax1.barh(comparison_df['Model'], comparison_df['RMSE'], color=colors_rmse, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('RMSE ($)', fontsize=11, fontweight='bold')
    ax1.set_title('RMSE Comparison (Lower is Better)', fontweight='bold', fontsize=13, pad=10)
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars1, comparison_df['RMSE'])):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'${val:.4f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 2. MAE Comparison
    ax2 = fig_main.add_subplot(gs[0, 1])
    colors_mae = ['#2ecc71' if i == best_mae_idx else '#e74c3c' for i in range(len(comparison_df))]
    bars2 = ax2.barh(comparison_df['Model'], comparison_df['MAE'], color=colors_mae, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('MAE ($)', fontsize=11, fontweight='bold')
    ax2.set_title('MAE Comparison (Lower is Better)', fontweight='bold', fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars2, comparison_df['MAE'])):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'${val:.4f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 3. R² Score Comparison
    ax3 = fig_main.add_subplot(gs[1, 0])
    colors_r2 = ['#2ecc71' if i == best_r2_idx else '#9b59b6' for i in range(len(comparison_df))]
    bars3 = ax3.barh(comparison_df['Model'], comparison_df['R²'], color=colors_r2, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('R² Score Comparison (Higher is Better)', fontweight='bold', fontsize=13, pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars3, comparison_df['R²'])):
        ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 4. MAPE Comparison
    ax4 = fig_main.add_subplot(gs[1, 1])
    colors_mape = ['#2ecc71' if i == best_mape_idx else '#f39c12' for i in range(len(comparison_df))]
    bars4 = ax4.barh(comparison_df['Model'], comparison_df['MAPE (%)'], color=colors_mape, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax4.set_title('MAPE Comparison (Lower is Better)', fontweight='bold', fontsize=13, pad=10)
    ax4.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars4, comparison_df['MAPE (%)'])):
        ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_main)
    plt.close()
    
    st.markdown("---")
    
    # Model insights
    st.subheader(" Model Insights")
    rankings = pd.DataFrame()
    rankings['Model'] = comparison_df['Model']
    rankings['RMSE_rank'] = comparison_df['RMSE'].rank()
    rankings['MAE_rank'] = comparison_df['MAE'].rank()
    rankings['R²_rank'] = comparison_df['R²'].rank(ascending=False)
    rankings['MAPE_rank'] = comparison_df['MAPE (%)'].rank()
    rankings['Avg_rank'] = rankings[['RMSE_rank', 'MAE_rank', 'R²_rank', 'MAPE_rank']].mean(axis=1)
    rankings = rankings.sort_values('Avg_rank')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(" Best Overall Model:")
        overall_best = rankings.iloc[0]['Model']
        overall_score = rankings.iloc[0]['Avg_rank']
        st.success(f"**{overall_best}** (Avg Rank: {overall_score:.2f})")
        
        st.markdown(" Performance Spread:")
        rmse_range = comparison_df['RMSE'].max() - comparison_df['RMSE'].min()
        st.info(f"RMSE Range: ${rmse_range:.4f}")
        
    with col2:
        metrics_normalized = comparison_df.copy()
        metrics_normalized['RMSE_norm'] = 1 - (metrics_normalized['RMSE'] - metrics_normalized['RMSE'].min()) / (metrics_normalized['RMSE'].max() - metrics_normalized['RMSE'].min() + 1e-10)
        metrics_normalized['MAE_norm'] = 1 - (metrics_normalized['MAE'] - metrics_normalized['MAE'].min()) / (metrics_normalized['MAE'].max() - metrics_normalized['MAE'].min() + 1e-10)
        metrics_normalized['R²_norm'] = (metrics_normalized['R²'] - metrics_normalized['R²'].min()) / (metrics_normalized['R²'].max() - metrics_normalized['R²'].min() + 1e-10)
        metrics_normalized['MAPE_norm'] = 1 - (metrics_normalized['MAPE (%)'] - metrics_normalized['MAPE (%)'].min()) / (metrics_normalized['MAPE (%)'].max() - metrics_normalized['MAPE (%)'].min() + 1e-10)
        
        st.markdown(" Most Consistent:")
        consistency_scores = []
        for idx, row in metrics_normalized.iterrows():
            values = [row['RMSE_norm'], row['MAE_norm'], row['R²_norm'], row['MAPE_norm']]
            consistency_scores.append(np.std(values))
        most_consistent_idx = np.argmin(consistency_scores)
        st.success(f"**{comparison_df.iloc[most_consistent_idx]['Model']}**")
        
        st.markdown(" R² Performance:")
        r2_range = comparison_df['R²'].max() - comparison_df['R²'].min()
        st.info(f"R² Range: {r2_range:.4f}")
    
    gc.collect()
    
    st.markdown("---")
    st.subheader("Best Model Configuration")
    
    best_model_idx = rankings.iloc[0].name
    best_model_info = st.session_state.model_results[best_model_idx]
    
    st.success(f"**Model: {best_model_info['model_name']}**")
    st.info(f"**Category: {best_model_info['config']['model_category']}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Metrics")
        st.markdown(f"- **RMSE**: ${best_model_info['metrics']['rmse']:.4f}")
        st.markdown(f"- **MAE**: ${best_model_info['metrics']['mae']:.4f}")
        st.markdown(f"- **R²**: {best_model_info['metrics']['r2']:.4f}")
        st.markdown(f"- **MAPE**: {best_model_info['metrics']['mape']:.2f}%")
    
    with col2:
        st.markdown("#### Hyperparameters")
        if 'hyperparameters' in best_model_info['config']:
            params = best_model_info['config']['hyperparameters']
            for key, value in params.items():
                st.markdown(f"- **{key}**: {value}")
        else:
            st.warning("Hyperparameters not stored for this model")
            st.markdown("**Features used:**")
            features = best_model_info['config'].get('features', [])
            if features:
                for feat in features:
                    st.markdown(f"- {feat}")
            else:
                st.markdown("- Price only")
    
    st.markdown("---")
    if st.button(" Back to Training", type="primary"):
        st.session_state.show_comparison = False
        st.rerun()
    
    st.stop()

# ==================== SIDEBAR ====================
st.sidebar.header("Configuration")

# Market Selection
st.sidebar.subheader("Market Selection")
market_choice = st.sidebar.radio("Select Market", ["Amman Stock Exchange (ASE)", "US Stock Market"])
st.sidebar.markdown("---")

# Stock Selection
st.sidebar.subheader("Stock Selection")

if market_choice == "Amman Stock Exchange (ASE)":
    sector_options = ["Select Sector..."] + list(sectors.keys())
    selected_sector = st.sidebar.selectbox("Sector", sector_options, key="sector_select")

    if selected_sector != "Select Sector...":
        subsector_options = ["Select Subsector..."] + list(sectors[selected_sector].keys())
        selected_subsector = st.sidebar.selectbox("Subsector", subsector_options, key="subsector_select")
        
        if selected_subsector != "Select Subsector...":
            company_options = ["Select Company..."] + sectors[selected_sector][selected_subsector]
            selected_company = st.sidebar.selectbox("Company", company_options, key="company_select")
            symbol = company_symbols.get(selected_company)
        else:
            selected_company = None
            symbol = None
    else:
        selected_subsector = None
        selected_company = None
        symbol = None
else:
    # US Market Selection
    us_sector_options = ["Select Sector..."] + list(us_sectors.keys())
    selected_us_sector = st.sidebar.selectbox("US Sector", us_sector_options, key="us_sector_select")

    if selected_us_sector != "Select Sector...":
        us_subsector_options = ["Select Subsector..."] + list(us_sectors[selected_us_sector].keys())
        selected_us_subsector = st.sidebar.selectbox("US Subsector", us_subsector_options, key="us_subsector_select")
        
        if selected_us_subsector != "Select Subsector...":
            us_company_options = ["Select Company..."] + us_sectors[selected_us_sector][selected_us_subsector]
            selected_company = st.sidebar.selectbox("US Company", us_company_options, key="us_company_select")
            symbol = us_symbols.get(selected_company)
        else:
            selected_company = None
            symbol = None
    else:
        selected_us_subsector = None
        selected_company = None
        symbol = None

# Only proceed if a company/symbol is selected
if selected_company and selected_company != "Select Company...":
    if not symbol:
        st.error(" Company symbol not found")
        st.stop()
    
    # Load data
    with st.spinner(f"Loading {selected_company}..."):
        df_full = load_data(symbol, market_choice)
    
    if df_full.empty:
        st.error(" No data available. Check the ticker symbol.")
        st.stop()
    
    df = df_full.copy()
   
    
    # إضافة عمود جديد يحسب نسبة التغير اليومية للسعر (Returns)
    price_col_temp = 'Closing' if 'Closing' in df.columns else 'Close'
    if price_col_temp in df.columns:
        df['Returns'] = df[price_col_temp].pct_change() * 100
        
    # حذف الصفوف الفارغة الناتجة عن العملية
    df = df.dropna(subset=['Returns'] if 'Returns' in df.columns else df.columns)
    # Model Category Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Model Category")
    model_category = st.sidebar.radio("Choose Model Type", 
                                    ["Deep Learning (LSTM/GRU/RNN)", 
                                        "Machine Learning (RF/XGBoost/etc)", 
                                        "Classical Time Series (ARIMA)"])
        
    # Model-specific parameters
    if model_category == "Deep Learning (LSTM/GRU/RNN)":
        st.sidebar.subheader("Model Selection")
        model_type = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU", "RNN", "Transformer"])
        st.sidebar.subheader("Model Architecture")
        hidden_dim = st.sidebar.slider("Hidden Dimension", min_value=32, max_value=256, value=64, step=32)
        num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=4, value=2)
        dropout = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.3, step=0.1)

        if model_type == "LSTM":
            bidirectional = st.sidebar.checkbox("Bidirectional LSTM", value=False)
            use_layernorm = st.sidebar.checkbox("Layer Normalization", value=False)
        else:
            bidirectional = False
            use_layernorm = False

        st.sidebar.subheader("Training Parameters")
        seq_len = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=60)
        epochs = st.sidebar.slider("Epochs", min_value=10, max_value=200, value=50)
        batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
        learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        patience = st.sidebar.slider("Early Stopping Patience", min_value=5, max_value=50, value=20)
    elif model_category == "Machine Learning (RF/XGBoost/etc)":
        st.sidebar.subheader("Model Selection")
        ml_model_type = st.sidebar.selectbox("Choose Model", [
            "Random Forest",
            "XGBoost", 
            "LightGBM",
            "CatBoost",
            "Gradient Boosting",
            "Linear Regression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "SVR"
        ])
        
        st.sidebar.subheader("Model Parameters")
        seq_len = st.sidebar.slider("Sequence Length", 10, 100, 60)
        
        if ml_model_type in ["Random Forest", "Gradient Boosting"]:
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50)
            max_depth = st.sidebar.slider("Max Depth", 3, 20, 10)
            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42}
        elif ml_model_type in ["XGBoost", "LightGBM", "CatBoost"]:
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50)
            learning_rate = st.sidebar.select_slider("Learning Rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
            max_depth = st.sidebar.slider("Max Depth", 3, 15, 6)
            model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'random_state': 42}
        elif ml_model_type in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.sidebar.slider("Alpha (Regularization)", 0.1, 10.0, 1.0, 0.1)
            model_params = {'alpha': alpha}
            if ml_model_type == "ElasticNet":
                l1_ratio = st.sidebar.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.1)
                model_params['l1_ratio'] = l1_ratio
        elif ml_model_type == "SVR":
            C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
            kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
            model_params = {'C': C, 'kernel': kernel}
        else:
            model_params = {}
    else:
        st.sidebar.subheader("ARIMA Model Selection")
        arima_model_type = st.sidebar.selectbox("Choose Model", 
                                                ["AR (AutoRegressive)", 
                                                 "MA (Moving Average)", 
                                                 "ARMA (AutoRegressive Moving Average)",
                                                 "ARIMA (AutoRegressive Integrated Moving Average)"])
        st.sidebar.subheader("ARIMA Parameters")
        if arima_model_type in ["AR (AutoRegressive)", "ARMA (AutoRegressive Moving Average)", "ARIMA (AutoRegressive Integrated Moving Average)"]:
            p_order = st.sidebar.slider("AR Order (p)", min_value=0, max_value=10, value=1)
        else:
            p_order = 0
        if arima_model_type == "ARIMA (AutoRegressive Integrated Moving Average)":
            d_order = st.sidebar.slider("Differencing Order (d)", min_value=0, max_value=2, value=1)
        else:
            d_order = 0
        if arima_model_type in ["MA (Moving Average)", "ARMA (AutoRegressive Moving Average)", "ARIMA (AutoRegressive Integrated Moving Average)"]:
            q_order = st.sidebar.slider("MA Order (q)", min_value=0, max_value=10, value=1)
        else:
            q_order = 0

    # Features
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Features")
    use_technical_indicators = st.sidebar.checkbox("Add Technical Indicators", value=False,
                                                    help="Add RSI, MACD, Bollinger Bands")

    pred_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

    train_button = st.sidebar.button(" Train Model & Predict", type="primary")

    # Model comparison
    if len(st.session_state.model_results) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader(" Trained Models")
        st.sidebar.markdown(f"**{len(st.session_state.model_results)}** models trained")
        if st.sidebar.button("Compare Models"):
            st.session_state.show_comparison = True
            st.rerun()
        if st.sidebar.button("Clear All Models"):
            st.session_state.model_results = []
            st.session_state.trained_models = {}
            st.rerun()

    # ==================== MAIN CONTENT ====================
    st.success(f" Data loaded successfully for **{selected_company}**!")
    
    st.subheader(" Data Preview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
    
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column selection
    st.subheader(" Select Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        date_column = st.selectbox("Date Column", options=df.columns.tolist(), 
                                  index=0 if 'Date' in df.columns else 0)
    with col2:
        price_column = st.selectbox("Price Column", options=df.columns.tolist(),
                                   index=df.columns.tolist().index('Closing') if 'Closing' in df.columns else 1)
    
    feature_columns = st.multiselect("Additional Features (optional)", 
                                    options=[col for col in df.columns if col not in [date_column, price_column]],
                                    default=[])
    
    # Data validation
    validation_issues = validate_data(df, date_column, price_column)
    if validation_issues:
        with st.expander("Data Quality Issues", expanded=True):
            for issue in validation_issues:
                st.warning(issue)
            
            if st.button("Clean Data"):
                df = df.dropna(subset=[date_column, price_column])
                df = df[df[price_column] > 0]
                st.session_state.data_cleaned = True
                validation_issues = validate_data(df, date_column, price_column)
                st.success("Data cleaned!")

    # Add technical indicators
    if use_technical_indicators:
        with st.spinner("Calculating technical indicators..."):
            df = calculate_technical_indicators(df, price_column)
            st.success("Technical indicators added!")
            feature_columns.extend(['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'])
    
    # ==================== DATA VISUALIZATION ====================
    st.subheader("Data Visualization")
    
    if date_column and price_column:
        viz_df = df.copy()
        viz_df = parse_dates(viz_df, date_column)
        if viz_df is not None:
            viz_df = viz_df.sort_values(date_column)
        
            tab1, tab2, tab3, tab4 = st.tabs([" Price Chart", " Statistical Analysis", " Distribution", " Correlation"])
            
            with tab1:
                st.markdown("### Price Over Time")
                has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
                
                if has_ohlc:
                    fig1 = plt.figure(figsize=(14, 7))
                    colors = ['green' if viz_df['Close'].iloc[i] >= viz_df['Open'].iloc[i] else 'red' 
                             for i in range(len(viz_df))]
                    for i in range(len(viz_df)):
                        plt.plot([viz_df[date_column].iloc[i], viz_df[date_column].iloc[i]], 
                                [viz_df['Low'].iloc[i], viz_df['High'].iloc[i]], 
                                color=colors[i], linewidth=0.5, alpha=0.5)
                    plt.plot(viz_df[date_column], viz_df['Close'], 
                            label='Close Price', linewidth=2, color='blue', alpha=0.7)
                    plt.fill_between(viz_df[date_column], viz_df['Low'], viz_df['High'], 
                                    alpha=0.1, color='gray', label='High-Low Range')
                    plt.title('Stock Price Chart (OHLC)', fontweight='bold', fontsize=14)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Price ($)', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                    
                    if 'Volume' in df.columns or 'Vluome' in df.columns:
                        vol_col = 'Volume' if 'Volume' in df.columns else 'Vluome'
                        fig_vol = plt.figure(figsize=(14, 4))
                        plt.bar(viz_df[date_column], viz_df[vol_col], color=colors, alpha=0.6, width=0.8)
                        plt.title('Trading Volume', fontweight='bold', fontsize=14)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Volume', fontsize=12)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_vol)
                        plt.close()
                else:
                    fig1 = plt.figure(figsize=(14, 7))
                    plt.plot(viz_df[date_column], viz_df[price_column], linewidth=2.5, color='blue', alpha=0.8)
                    plt.fill_between(viz_df[date_column], viz_df[price_column], alpha=0.2, color='blue')
                    plt.title(f'{price_column} Over Time', fontweight='bold', fontsize=14)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Price ($)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                gc.collect()
            
            with tab2:
                st.markdown("### Statistical Summary")
                price_data = viz_df[price_column].dropna()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"${price_data.mean():.2f}")
                    st.metric("Median", f"${price_data.median():.2f}")
                with col2:
                    st.metric("Std Dev", f"${price_data.std():.2f}")
                    st.metric("Variance", f"${price_data.var():.2f}")
                with col3:
                    st.metric("Min", f"${price_data.min():.2f}")
                    st.metric("Max", f"${price_data.max():.2f}")
                with col4:
                    st.metric("Range", f"${price_data.max() - price_data.min():.2f}")
                    st.metric("Skewness", f"{price_data.skew():.2f}")
                
                st.markdown("### Moving Averages Analysis")
                ma_periods = [7, 21, 50]
                fig_ma = plt.figure(figsize=(14, 7))
                plt.plot(viz_df[date_column], viz_df[price_column], label='Price', linewidth=2, color='black', alpha=0.7)
                colors_ma = ['blue', 'orange', 'green']
                for period, color in zip(ma_periods, colors_ma):
                    if len(viz_df) >= period:
                        ma = viz_df[price_column].rolling(window=period).mean()
                        plt.plot(viz_df[date_column], ma, label=f'{period}-Day MA', linewidth=2, linestyle='--', color=color, alpha=0.7)
                plt.title('Price with Moving Averages', fontweight='bold', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Price ($)', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_ma)
                plt.close()
                
                if len(price_data) > 1:
                    st.markdown("### Daily Returns Analysis")
                    returns = price_data.pct_change().dropna() * 100
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Daily Return", f"{returns.mean():.2f}%")
                        st.metric("Volatility (Std)", f"{returns.std():.2f}%")
                    with col2:
                        st.metric("Best Day", f"{returns.max():.2f}%")
                        st.metric("Worst Day", f"{returns.min():.2f}%")
                    
                    fig_ret = plt.figure(figsize=(14, 5))
                    plt.plot(returns.values, linewidth=1.5, color='purple', alpha=0.7)
                    plt.axhline(0, color='red', linestyle='--', linewidth=1)
                    plt.fill_between(range(len(returns)), returns.values, 0, where=(returns.values > 0), alpha=0.3, color='green', label='Gains')
                    plt.fill_between(range(len(returns)), returns.values, 0, where=(returns.values < 0), alpha=0.3, color='red', label='Losses')
                    plt.title('Daily Returns (%)', fontweight='bold', fontsize=14)
                    plt.xlabel('Trading Days', fontsize=12)
                    plt.ylabel('Return (%)', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_ret)
                    plt.close()
                gc.collect()
            
            with tab3:
                st.markdown("### Price Distribution")
                price_data_clean = viz_df[price_column].dropna()
                fig_dist = plt.figure(figsize=(14, 10))
                gs = fig_dist.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                ax1 = fig_dist.add_subplot(gs[0, 0])
                ax1.hist(price_data_clean, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax1.axvline(price_data_clean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${price_data_clean.mean():.2f}')
                ax1.axvline(price_data_clean.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${price_data_clean.median():.2f}')
                ax1.set_title('Price Distribution (Histogram)', fontweight='bold')
                ax1.set_xlabel('Price ($)')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig_dist.add_subplot(gs[0, 1])
                box = ax2.boxplot(price_data_clean, vert=True, patch_artist=True)
                box['boxes'][0].set_facecolor('lightblue')
                ax2.set_title('Price Box Plot', fontweight='bold')
                ax2.set_ylabel('Price ($)')
                ax2.grid(True, alpha=0.3, axis='y')
                
                ax3 = fig_dist.add_subplot(gs[1, 0])
                sorted_data = np.sort(price_data_clean)
                cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
                ax3.plot(sorted_data, cumulative, linewidth=2, color='purple')
                ax3.set_title('Cumulative Distribution', fontweight='bold')
                ax3.set_xlabel('Price ($)')
                ax3.set_ylabel('Cumulative %')
                ax3.grid(True, alpha=0.3)
                
                ax4 = fig_dist.add_subplot(gs[1, 1])
                stats.probplot(price_data_clean, dist="norm", plot=ax4)
                ax4.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_dist)
                plt.close(fig_dist)
                gc.collect()
            
            with tab4:
                st.markdown("### Correlation Analysis")
                numeric_cols = viz_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    corr_matrix = viz_df[numeric_cols].corr()
                    fig_corr = plt.figure(figsize=(10, 8))
                    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    plt.colorbar(im, label='Correlation Coefficient')
                    for i in range(len(corr_matrix)):
                        for j in range(len(corr_matrix)):
                            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10)
                    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
                    plt.yticks(range(len(numeric_cols)), numeric_cols)
                    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close()
                    
                    if price_column in numeric_cols:
                        st.markdown(f"### Correlation with {price_column}")
                        target_corr = corr_matrix[price_column].drop(price_column).sort_values(ascending=False)
                        fig_target = plt.figure(figsize=(10, 6))
                        colors_corr = ['green' if x > 0 else 'red' for x in target_corr.values]
                        plt.barh(range(len(target_corr)), target_corr.values, color=colors_corr, alpha=0.7)
                        plt.yticks(range(len(target_corr)), target_corr.index)
                        plt.xlabel('Correlation Coefficient', fontsize=12)
                        plt.title(f'Features Correlation with {price_column}', fontweight='bold', fontsize=14)
                        plt.axvline(0, color='black', linewidth=1)
                        plt.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig_target)
                        plt.close()
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis")
                gc.collect()
    
    # ==================== MODEL TRAINING ====================
    if train_button:
        set_seed(62)
        try:
            with st.spinner("Preparing data..."):
                df = parse_dates(df, date_column)
                if df is None:
                    st.stop()
                
                df = df.sort_values(date_column)
                df = df.dropna(subset=[price_column])
                
                if feature_columns:
                  # حذفنا bfill واستخدمنا dropna لمنع الموديل من رؤية المستقبل
                    data = df[[price_column] + feature_columns].fillna(method='ffill').dropna().values
                else:
                    data = df[[price_column]].values
                
                if model_category == "Deep Learning (LSTM/GRU/RNN)":
                    if len(data) < seq_len + 10:
                        st.error(f" Not enough data! Need at least {seq_len + 10} rows.")
                        st.stop()
                    
                    input_dim = data.shape[1]
                    
                    if model_type == "LSTM":
                        model = LSTMModel(input_dim, hidden_dim, num_layers, dropout, bidirectional, 1, use_layernorm)
                    elif model_type == "GRU":
                        model = GRUModel(input_dim, hidden_dim, num_layers, dropout)
                    elif model_type == "Transformer":
                        model = TransformerModel(input_dim, hidden_dim, num_layers, dropout, num_heads=4)
                    else:
                        model = RNNModel(input_dim, hidden_dim, num_layers, dropout)

                    st.subheader("Model Architecture")
                    total_params = sum(p.numel() for p in model.parameters())
                    st.markdown(f"**Total Parameters**: {total_params:,}")
                    
                    st.subheader(" Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    trainer = TimeSeriesTrainer(
                        model, seq_len, batch_size, epochs, learning_rate,
                        early_stopping=True, patience=patience
                    )
                    results = trainer.train(data, progress_bar, status_text)
                    model_name = model_type
                elif model_category == "Machine Learning (RF/XGBoost/etc)":
                    if len(data) < seq_len + 10:
                        st.error(f"Not enough data! Need at least {seq_len + 10} rows.")
                        st.stop()
                    
                    st.subheader(" Model Configuration")
                    st.markdown(f"**Model**: {ml_model_type}")
                    st.markdown(f"**Parameters**: {model_params}")
                    
                    st.subheader(" Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    trainer = MLTrainer(model_type=ml_model_type, seq_len=seq_len, **model_params)
                    results = trainer.train(data, progress_bar, status_text)
                    model_name = ml_model_type
                    
                    st.markdown("---")
                    st.subheader("Final Validation Check")
                    y_true = results['y_true'].flatten()
                    y_pred = results['y_pred'].flatten()
                    
                    naive_pred = np.roll(y_true, 1)[1:]
                    naive_mae = mean_absolute_error(y_true[1:], naive_pred)
                    model_mae = results['mae']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Naive MAE (yesterday's price)", f"${naive_mae:.4f}")
                    with col2:
                        st.metric("Model MAE", f"${model_mae:.4f}", 
                                delta=f"{((naive_mae - model_mae) / naive_mae * 100):.1f}% better" if model_mae < naive_mae else None,
                                delta_color="normal")
                    
                    if model_mae < naive_mae:
                        st.success("Model beats naive baseline!")
                    else:
                        st.warning("Model doesn't beat naive - possible issue")
                    
                    shift_corr = np.corrcoef(y_true[:-1], y_pred[1:])[0, 1]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Shift Correlation", f"{shift_corr:.4f}")
                        st.caption("Should be < 0.90 for true prediction")
                    with col2:
                        if shift_corr > 0.95:
                            st.error("Model might be just shifting data!")
                        else:
                            st.success("Model makes real predictions!")
                    
                    diffs = np.abs(y_true - y_pred)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Abs Diff", f"${diffs.mean():.4f}")
                    with col2:
                        st.metric("Max Diff", f"${diffs.max():.4f}")
                    with col3:
                        st.metric("Std of Diffs", f"${diffs.std():.4f}")
                    
                    st.write("**Last 10 Predictions:**")
                    comparison = pd.DataFrame({
                        'Actual': y_true[-10:].round(4),
                        'Predicted': y_pred[-10:].round(4),
                        'Error': (y_true[-10:] - y_pred[-10:]).round(4),
                        'Error %': ((y_true[-10:] - y_pred[-10:]) / y_true[-10:] * 100).round(2)
                    })
                    st.dataframe(comparison, use_container_width=True)
                    
                    avg_error_pct = np.abs((y_true - y_pred) / y_true * 100).mean()
                    if avg_error_pct < 1.0:
                        st.success(f"Average error: {avg_error_pct:.2f}% - Excellent!")
                    elif avg_error_pct < 3.0:
                        st.info(f"Average error: {avg_error_pct:.2f}% - Good")
                    else:
                        st.warning(f"Average error: {avg_error_pct:.2f}% - Needs improvement")
                    st.markdown("---")
                    
                else:  # ARIMA
                    if len(data) < 50:
                        st.error(" Not enough data! Need at least 50 rows for ARIMA.")
                        st.stop()
                    
                    arima_order = (p_order, d_order, q_order)
                    if arima_model_type == "AR (AutoRegressive)": model_name = "AR"
                    elif arima_model_type == "MA (Moving Average)": model_name = "MA"
                    elif arima_model_type == "ARMA (AutoRegressive Moving Average)": model_name = "ARMA"
                    else: model_name = "ARIMA"
                    
                    st.subheader(" Model Configuration")
                    st.markdown(f"**Order (p, d, q)**: {arima_order}")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    trainer = ARIMATrainer(order=arima_order, model_type=model_name)
                    results = trainer.train(data, progress_bar, status_text)
                
                progress_bar.progress(1.0)
                status_text.text(" Training completed!")
                
                hyperparams = {}
                if model_category == "Deep Learning (LSTM/GRU/RNN)":
                    hyperparams = {
                        'Model Type': model_type, 'Hidden Dimension': hidden_dim, 'Number of Layers': num_layers,
                        'Dropout': dropout, 'Sequence Length': seq_len, 'Epochs': epochs, 'Batch Size': batch_size,
                        'Learning Rate': learning_rate, 'Patience': patience
                    }
                    if model_type == "LSTM":
                        hyperparams['Bidirectional'] = bidirectional
                        hyperparams['Layer Normalization'] = use_layernorm
                elif model_category == "Machine Learning (RF/XGBoost/etc)":
                    hyperparams = {'Model Type': ml_model_type, 'Sequence Length': seq_len}
                    for key, value in model_params.items():
                        hyperparams[key.replace('_', ' ').title()] = value
                else:  
                    hyperparams = {'Model Type': arima_model_type, 'AR Order (p)': p_order, 'Differencing (d)': d_order, 'MA Order (q)': q_order}

                st.session_state.model_results.append({
                    'model_name': model_name, 'metrics': results, 'timestamp': datetime.now(),
                    'config': {'model_category': model_category, 'features': feature_columns, 'hyperparameters': hyperparams}
                })
                
                st.subheader(" Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"${results['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"${results['mae']:.2f}")
                with col3:
                    st.metric("R²", f"{results['r2']:.4f}")
                with col4:
                    st.metric("MAPE", f"{results['mape']:.2f}%")
                
                if 'train_losses' in results and 'val_losses' in results:
                    fig_loss = plt.figure(figsize=(12, 5))
                    plt.plot(results['train_losses'], label='Training Loss', linewidth=2, color='blue', marker='o', markersize=3, alpha=0.8)
                    plt.plot(results['val_losses'], label='Validation Loss', linewidth=2, color='red', marker='s', markersize=3, alpha=0.8)
                    plt.title('Training & Validation Loss Over Time', fontweight='bold', fontsize=14)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('Loss (MSE)', fontsize=12)
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_loss)
                    plt.close()
                
                st.subheader(" Model Validation Performance")
                y_true = results['y_true'].flatten()
                y_pred = results['y_pred'].flatten()
                n = len(y_true)
                residuals = y_true - y_pred
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs(residuals / np.clip(y_true, 1e-8, None))) * 100
                
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(range(n), y_true, label='Actual', linewidth=2, color='black', alpha=0.8)
                ax1.plot(range(n), y_pred, label='Predicted', linewidth=1.5, linestyle='--', color='red', alpha=0.7)
                ax1.fill_between(range(n), y_true, y_pred, alpha=0.2, color='red')
                ax1.set_title(f'{model_name} Model - Full Forecast | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%', fontweight='bold', fontsize=12)
                ax1.set_ylabel('Close Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(gs[1, :])
                zoom_n = min(120, n)
                ax2.plot(range(zoom_n), y_true[-zoom_n:], label='Actual', linewidth=2, color='black', marker='o', markersize=3)
                ax2.plot(range(zoom_n), y_pred[-zoom_n:], label='Predicted', linewidth=1.5, linestyle='--', color='red', marker='s', markersize=3)
                ax2.set_title(f'Zoomed View - Last {zoom_n} Points', fontweight='bold')
                ax2.set_ylabel('Close Price')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                ax3 = fig.add_subplot(gs[2, 0])
                ax3.bar(range(n), residuals, alpha=0.6, color='steelblue', width=1.0)
                ax3.axhline(0, color='red', linestyle='--', linewidth=2)
                ax3.set_title('Residuals (Prediction Errors)', fontweight='bold')
                ax3.set_ylabel('Residual')
                ax3.grid(True, alpha=0.3, axis='y')
                
                ax4 = fig.add_subplot(gs[2, 1])
                ax4.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
                min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                ax4.set_xlabel('Actual')
                ax4.set_ylabel('Predicted')
                ax4.set_title('Actual vs Predicted', fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                gc.collect()
                
                st.markdown(f"""
                ###  Performance Summary
                
                | Metric | Value |
                |--------|-------|
                | MAE | {mae:.4f} |
                | RMSE | {rmse:.4f} |
                | MAPE | {mape:.2f}% |
                | Mean Residual | {np.mean(residuals):.4f} |
                | Std Residual | {np.std(residuals):.4f} |
                """)
                
                st.subheader(" Future Price Predictions")
                if model_category == "Deep Learning (LSTM/GRU/RNN)":
                    last_sequence = data[-seq_len:]
                else:
                    last_sequence = data
                
                future_preds = trainer.predict(last_sequence, pred_days)
                last_date = df[date_column].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days, freq='D')
                
                lookback_days = min(100, len(df))
                historical = df[price_column].values[-lookback_days:]
                hist_dates = df[date_column].iloc[-lookback_days:]
                
                fig_future, ax_future = plt.subplots(figsize=(14, 7))
                ax_future.plot(hist_dates, historical, label='Historical', linewidth=2.5, color='blue', alpha=0.8)
                ax_future.plot(future_dates, future_preds, label='Predicted', linewidth=2.5, linestyle='--', color='red', marker='o', markersize=8, alpha=0.8)
                ax_future.axvline(x=last_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Today')
                ax_future.set_title(f'Price Forecast - Next {pred_days} Days', fontweight='bold', fontsize=14)
                ax_future.set_xlabel('Date', fontsize=12)
                ax_future.set_ylabel('Price ($)', fontsize=12)
                ax_future.legend(fontsize=10)
                ax_future.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_future)
                plt.close(fig_future)
                gc.collect()
                
                st.success(f" Model trained successfully!")
                
        except ValueError as e:
            st.error(f" Data error: {str(e)}")
        except RuntimeError as e:
            st.error(f" Model error: {str(e)}")
        except Exception as e:
            st.error(f" Unexpected error: {str(e)}")
            st.exception(e)

else:
    st.markdown("""
Features:

Machine Learning Models:
* XGBoost: Extreme Gradient Boosting
* Random Forest: Ensemble of decision trees
* LightGBM: Light Gradient Boosting Machine
* CatBoost: Categorical Boosting
* Gradient Boosting: Traditional gradient boosting
* Linear Regression: Basic linear model
* Ridge: L2 regularized regression
* Lasso: L1 regularized regression
* ElasticNet: Combined L1/L2 regularization
* SVR: Support Vector Regression

Deep Learning Models:
* LSTM: Long Short-Term Memory with bidirectional & layer normalization
* GRU: Gated Recurrent Unit
* RNN: Basic Recurrent Neural Network

Classical Time Series Models:
* AR: AutoRegressive
* MA: Moving Average
* ARMA: AutoRegressive Moving Average
* ARIMA: AutoRegressive Integrated Moving Average

Disclaimer:
Educational purposes only. Not financial advice.
    """)

st.markdown("---")