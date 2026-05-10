import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
from streamlit_autorefresh import st_autorefresh

# --- 0. 檔案與初始化 ---
DB_FILE = "scan_history.csv"
if 'search_history' not in st.session_state: st.session_state['search_history'] = []
if 'scan_index' not in st.session_state: st.session_state['scan_index'] = 0

def load_history():
    if os.path.exists(DB_FILE):
        try: return pd.read_csv(DB_FILE)
        except: return pd.DataFrame(columns=["觸發時間", "頻率", "板塊", "代碼", "股票名稱", "收盤價", "觸發策略"])
    return pd.DataFrame(columns=["觸發時間", "頻率", "板塊", "代碼", "股票名稱", "收盤價", "觸發策略"])

if 'scan_log' not in st.session_state: st.session_state['scan_log'] = load_history()

def save_record_to_csv(new_df):
    old_df = load_history()
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["觸發時間", "頻率", "代碼", "觸發策略"])
    combined.to_csv(DB_FILE, index=False, encoding='utf-8-sig')

TW_STOCKS = {
    "✍️ 自訂輸入": "", "2330.TW - 台積電": "2330.TW", "2317.TW - 鴻海": "2317.TW", 
    "2454.TW - 聯發科": "2454.TW", "NVDA - 輝達": "NVDA", "TSLA - 特斯拉": "TSLA"
}

SCAN_POOLS = {
    "🏆 權值核心": {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達", "2881.TW": "富邦金"},
    "💻 AI 半導體": {"3231.TW": "緯創", "2356.TW": "英業達", "2376.TW": "技嘉", "3037.TW": "欣興", "6669.TW": "緯穎"},
    "🏦 金融族群": {"2886.TW": "兆豐金", "2884.TW": "玉山金", "2892.TW": "第一金", "2885.TW": "元大金", "5880.TW": "合庫金"},
    "🚢 航運艦隊": {"2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", 
        "2637.TW": "慧洋-KY", "2606.TW": "裕民", 
        "2618.TW": "長榮航", "2610.TW": "華航", "2646.TW": "星宇航空"},
    "🔌 電子零組件": {"3017.TW": "奇鋐", "3324.TW": "雙鴻", "3653.TW": "健策", "2421.TW": "建準",
        "3037.TW": "欣興", "2368.TW": "金像電", "2313.TW": "華通", "8046.TW": "南電",
        "2327.TW": "國巨", "2492.TW": "華新科", 
        "3533.TW": "嘉澤", "3023.TW": "信邦", "2392.TW": "正崴"},
    "🟢 PCB 電子之母": {"2383.TW": "台光電", "6274.TW": "台燿", "3037.TW": "欣興", "8046.TW": "南電", "2368.TW": "金像電", "2313.TW": "華通", "4958.TW": "臻鼎-KY",
        "2355.TW": "敬鵬", "2367.TW": "燿華", "6269.TW": "台郡"},
    "💰 熱門 ETF": {"0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00929.TW": "復華台灣科技優息", "00713.TW": "元大台灣高息低波"}
}

STRATEGIES = [
    "均線黃金交叉 (20MA & 50MA)", "RSI 超買超賣 (30買/70賣)", 
    "MACD 黃金交叉/死亡交叉", "纏論核心 (底背離+二買策略)",
    "布林通道+RSI反轉"
]

# --- 數據處理與圖表 ---
def get_latest_price(ticker_str):
    try:
        tkr = yf.Ticker(ticker_str)
        fast = tkr.fast_info
        return fast.last_price, (fast.last_price - fast.previous_close), ((fast.last_price - fast.previous_close)/fast.previous_close*100)
    except: return None, None, None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, start, end, interval="1d"):
    return yf.download(ticker, start=start, end=end, interval=interval, progress=False)

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'], df['Lower'] = df['MA20']+(df['std']*2), df['MA20']-(df['std']*2)
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

# --- 核心策略大腦 (乾淨隔離版) ---
def generate_signals(df, strategy_choice):
    df_bt = df.copy()
    df_bt['Signal'] = np.nan 
    
    if strategy_choice == "均線黃金交叉 (20MA & 50MA)":
        df_bt.loc[df_bt['SMA_20'] > df_bt['SMA_50'], 'Signal'] = 1
        df_bt.loc[df_bt['SMA_20'] <= df_bt['SMA_50'], 'Signal'] = 0
        
    elif strategy_choice == "RSI 超買超賣 (30買/70賣)":
        df_bt.loc[df_bt['RSI'] < 30, 'Signal'] = 1
        df_bt.loc[df_bt['RSI'] > 70, 'Signal'] = 0
        
    elif strategy_choice == "MACD 黃金交叉/死亡交叉":
        df_bt.loc[df_bt['MACD'] > df_bt['Signal_Line'], 'Signal'] = 1
        df_bt.loc[df_bt['MACD'] <= df_bt['Signal_Line'], 'Signal'] = 0
        
    elif strategy_choice == "纏論核心 (底背離+二買策略)":
        window = 30
        low_lookback = df_bt['Low'].rolling(window=window).min()
        hist_min = df_bt['MACD_Hist'].rolling(window=window).min()
        div_buy = (df_bt['Low'] <= low_lookback) & (df_bt['MACD_Hist'] > hist_min.shift(1))
        
        last_low_ref = df_bt['Low'].rolling(window=window*2).min().shift(5)
        sec_buy = (df_bt['Low'] > last_low_ref) & (df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['MACD'].shift(1) < df_bt['Signal_Line'].shift(1))
        
        df_bt.loc[div_buy | sec_buy, 'Signal'] = 1
        df_bt.loc[df_bt['MACD'] < df_bt['Signal_Line'], 'Signal'] = 0
        
    elif strategy_choice == "布林通道+RSI反轉":
        df_bt.loc[(df_bt['Low'] <= df_bt['Lower']) & (df_bt['RSI'] < 35), 'Signal'] = 1
        df_bt.loc[(df_bt['High'] >= df_bt['Upper']) | (df_bt['RSI'] > 70), 'Signal'] = 0
    
    df_bt['Signal'] = df_bt['Signal'].ffill().fillna(0)
    return df_bt

# --- 介面開始 ---
st.set_page_config(page_title="AI 股票戰艦 v18.3", layout="wide")
app_mode = st.sidebar.radio("模式切換", ["🔍 單股深度分析", "🚀 雷達掃描"])

if app_mode == "🔍 單股深度分析":
    st.title("📊 股票深度分析 (完整圖表版)")
    st.sidebar.header("1. 分析設定")
    sel = st.sidebar.selectbox("快速選擇股票", list(TW_STOCKS.keys()))
    ticker = st.sidebar.text_input("代碼", value="2330.TW") if sel == "✍️ 自訂輸入" else TW_STOCKS[sel]
    
    interval_choice = st.sidebar.selectbox("K線頻率 (Timeframe)", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
    inv_map = {"日K (1d)": "1d", "15分K (15m)": "15m", "5分K (5m)": "5m"}
    interval_val = inv_map[interval_choice]

    p_map = {"3年": 1095, "1年": 365, "6個月": 180, "3個月": 90, "1個月": 30, "20日": 20, "10日": 10, "5日": 5, "1日": 1}
    default_index = 4 if interval_val == "1d" else 7 
    p_sel = st.sidebar.selectbox("查詢期間", list(p_map.keys()), index=default_index)
    
    end_d = datetime.now()
    start_d = end_d - timedelta(days=p_map[p_sel])

    if interval_val in ["5m", "15m"] and (end_d - start_d).days > 59:
        start_d = end_d - timedelta(days=59)
        st.sidebar.warning("⚠️ 分K線最多支援近 60 天資料，已修正開始日期。")

    st.sidebar.markdown("---")
    strat = st.sidebar.selectbox("交易策略", STRATEGIES)

    if st.sidebar.button("開始分析"):
        with st.spinner('運算中...'):
            cp, ch, cpct = get_latest_price(ticker)
            if cp: st.metric(f"⚡ {ticker.upper()} 最新報價", f"{cp:.2f}", f"{ch:.2f} ({cpct:.2f}%)")
            
            df = fetch_stock_data(ticker, start_d, end_d, interval=interval_val)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                df_bt = generate_signals(df, strat)
                
                df_bt['Action_Buy'] = (df_bt['Signal'] == 1) & (df_bt['Signal'].shift(1) == 0)
                df_bt['Action_Sell'] = (df_bt['Signal'] == 0) & (df_bt['Signal'].shift(1) == 1)
                
                sig_now = df_bt['Signal'].iloc[-1]
                if sig_now == 1: st.success(f"🤖 目前狀態：【買進/持有】(策略：{strat})")
                else: st.warning(f"🤖 目前狀態：【觀望/賣出】(策略：{strat})")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(173, 204, 255, 0.2)'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(173, 204, 255, 0.2)'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)', name='布林通道'), row=1, col=1)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
                
                buy_pts = df_bt[df_bt['Action_Buy']]; sell_pts = df_bt[df_bt['Action_Sell']]
                fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=15), name='買入'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='#FF4B4B', size=15), name='賣出'), row=1, col=1)
                
                vol_colors = ['#d62728' if row['Close'] < row['Open'] else '#2ca02c' for idx, row in df.iterrows()]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='成交量'), row=2, col=1)
                
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, title=f"{ticker.upper()} 量價走勢圖 ({interval_choice})")
                st.plotly_chart(fig, use_container_width=True)

# 💡 以下就是被我不小心隱藏掉的完整雷達模塊，現在已全數補回！
elif app_mode == "🚀 雷達掃描":
    st.title("🚀 AI 自動選股雷達 (多週期掃描版)")
    st_autorefresh(interval=300000, key="fscancounter")
    st.sidebar.header("🎯 巡航設定")
    
    scan_interval_choice = st.sidebar.selectbox("雷達掃描 K線頻率", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
    inv_map = {"日K (1d)": "1d", "15分K (15m)": "15m", "5分K (5m)": "5m"}
    scan_interval_val = inv_map[scan_interval_choice]

    selected_strats = st.sidebar.multiselect("策略多選", STRATEGIES, default=["纏論核心 (底背離+二買策略)"])
    is_auto = st.sidebar.toggle("啟用自動輪巡", value=True)
    categories = list(SCAN_POOLS.keys())
    
    if is_auto:
        current_cat = categories[st.session_state['scan_index']]
        st.session_state['scan_index'] = (st.session_state['scan_index'] + 1) % len(categories)
    else:
        current_cat = st.sidebar.selectbox("指定板塊", categories)
        
    st.subheader(f"📡 目前掃描：【{current_cat}】 | 頻率：{scan_interval_choice}")
    buy_candidates = []
    end_d = datetime.now()
    start_d = end_d - timedelta(days=59 if scan_interval_val in ["5m", "15m"] else 180)
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    progress = st.progress(0)
    status = st.empty()
    stocks = SCAN_POOLS[current_cat]
    
    for idx, (ticker, name) in enumerate(stocks.items()):
        status.text(f"掃描中: {name}...")
        try:
            time.sleep(0.15)
            df = fetch_stock_data(ticker, start_d, end_d, interval=scan_interval_val)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                for strat in selected_strats:
                    df_sig = generate_signals(df, strat)
                    # 抓取最後一根K線是否為買進持有狀態
                    if df_sig['Signal'].iloc[-1] == 1:
                        price = round(df_sig['Close'].iloc[-1], 2)
                        res = {
                            "觸發時間": current_time_str, 
                            "頻率": scan_interval_choice,
                            "板塊": current_cat, "代碼": ticker, "股票名稱": name, 
                            "收盤價": price, "觸發策略": strat
                        }
                        buy_candidates.append(res)
                        save_record_to_csv(pd.DataFrame([res]))
        except: pass
        progress.progress((idx + 1) / len(stocks))
    
    st.session_state['scan_log'] = load_history()
    
    if buy_candidates:
        st.success(f"🔥 發現 {len(buy_candidates)} 個訊號！")
        st.dataframe(pd.DataFrame(buy_candidates), use_container_width=True)
    else:
        st.warning(f"⚠️ 本次掃描無符合訊號。")
    
    st.markdown("---")
    st.subheader("💾 歷史掃描總表")
    if not st.session_state['scan_log'].empty:
        log_display = st.session_state['scan_log'].sort_values(by="觸發時間", ascending=False)
        st.dataframe(log_display, use_container_width=True, height=400)
        st.download_button("📥 下載存檔", log_display.to_csv(index=False).encode('utf-8-sig'), "stock_history.csv", "text/csv")
        if st.button("🗑️ 清空所有歷史"):
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.session_state['scan_log'] = pd.DataFrame(columns=["觸發時間", "頻率", "板塊", "代碼", "股票名稱", "收盤價", "觸發策略"])
            st.rerun()