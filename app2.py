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

# --- 檔案儲存設定 ---
DB_FILE = "scan_history.csv"

# --- 0. 初始化狀態與載入歷史紀錄 ---
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'scan_index' not in st.session_state:
    st.session_state['scan_index'] = 0

def load_history():
    if os.path.exists(DB_FILE):
        try: return pd.read_csv(DB_FILE)
        except: return pd.DataFrame(columns=["觸發時間", "頻率", "板塊", "代碼", "股票名稱", "收盤價", "觸發策略"])
    else:
        return pd.DataFrame(columns=["觸發時間", "頻率", "板塊", "代碼", "股票名稱", "收盤價", "觸發策略"])

if 'scan_log' not in st.session_state:
    st.session_state['scan_log'] = load_history()

def save_record_to_csv(new_df):
    if os.path.exists(DB_FILE):
        old_df = pd.read_csv(DB_FILE)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["觸發時間", "頻率", "代碼", "觸發策略"])
        combined.to_csv(DB_FILE, index=False, encoding='utf-8-sig')
    else:
        new_df.to_csv(DB_FILE, index=False, encoding='utf-8-sig')

# --- 股票清單與策略 ---
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
    "纏論簡化版 (MACD 底背馳)", "布林通道+RSI反轉"
]

# --- 核心運算函數 ---
def get_latest_price(ticker_str):
    try:
        tkr = yf.Ticker(ticker_str)
        fast = tkr.fast_info
        return fast.last_price, (fast.last_price - fast.previous_close), ((fast.last_price - fast.previous_close)/fast.previous_close*100)
    except: return None, None, None

# 💡 升級：加入 interval 參數支援分K
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, start, end, interval="1d"):
    return yf.download(ticker, start=start, end=end, interval=interval, progress=False)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['std'] * 2)
    df['Lower'] = df['MA20'] - (df['std'] * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

def generate_signals(df, strategy_choice):
    df_bt = df.copy()
    df_bt['Signal'] = 0
    if strategy_choice == "均線黃金交叉 (20MA & 50MA)":
        df_bt.loc[df_bt['SMA_20'] > df_bt['SMA_50'], 'Signal'] = 1
    elif strategy_choice == "RSI 超買超賣 (30買/70賣)":
        df_bt['Signal'] = np.nan
        df_bt.loc[df_bt['RSI'] < 30, 'Signal'] = 1
        df_bt.loc[df_bt['RSI'] > 70, 'Signal'] = 0
        df_bt['Signal'] = df_bt['Signal'].ffill().fillna(0)
    elif strategy_choice == "MACD 黃金交叉/死亡交叉":
        df_bt.loc[df_bt['MACD'] > df_bt['Signal_Line'], 'Signal'] = 1
    elif strategy_choice == "纏論核心 (底背離+二買策略)":
        low_20 = df_bt['Low'].rolling(window=20).min()
        hist_min_20 = df_bt['MACD_Hist'].rolling(window=20).min()
        div_buy = (df_bt['Low'] <= low_20) & (df_bt['MACD_Hist'] > hist_min_20.shift(1))
        last_low = df_bt['Low'].rolling(window=30).min().shift(5)
        sec_buy = (df_bt['Low'] > last_low) & (df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['MACD'].shift(1) < df_bt['Signal_Line'].shift(1))
        df_bt.loc[div_buy | sec_buy, 'Signal'] = 1
    elif strategy_choice == "纏論簡化版 (MACD 底背馳)":
        low_20 = df_bt['Low'].rolling(window=20).min()
        hist_min_20 = df_bt['MACD_Hist'].rolling(window=20).min()
        df_bt.loc[(df_bt['Low'] <= low_20) & (df_bt['MACD_Hist'] > hist_min_20.shift(1)), 'Signal'] = 1
    elif strategy_choice == "布林通道+RSI反轉":
        df_bt.loc[(df_bt['Low'] <= df_bt['Lower']) & (df_bt['RSI'] < 35), 'Signal'] = 1
    
    df_bt.loc[df_bt['MACD'] < df_bt['Signal_Line'], 'Signal'] = 0
    df_bt['Signal'] = df_bt['Signal'].replace(0, np.nan).ffill().fillna(0)
    return df_bt

def run_backtest(df, strategy_choice, sl_pct, tp_pct, initial_capital=100000):
    df_bt = generate_signals(df.copy(), strategy_choice)
    df_bt['Position'] = 0; df_bt['Action_Buy'] = False; df_bt['Action_Sell'] = False
    pos, entry = 0, 0.0
    for i in range(1, len(df_bt)):
        curr = df_bt['Close'].iloc[i]
        sig = df_bt['Signal'].iloc[i-1]
        if pos == 1:
            if curr <= entry*(1-sl_pct) or curr >= entry*(1+tp_pct) or sig == 0:
                pos = 0; df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
        elif pos == 0 and sig == 1:
            pos, entry = 1, curr; df_bt.iloc[i, df_bt.columns.get_loc('Action_Buy')] = True
        df_bt.iloc[i, df_bt.columns.get_loc('Position')] = pos
    df_bt['Strategy_Value'] = initial_capital * (1 + (df_bt['Position'].shift(1).fillna(0) * df_bt['Close'].pct_change())).cumprod()
    df_bt['Market_Value'] = initial_capital * (1 + df_bt['Close'].pct_change()).cumprod()
    return df_bt.fillna(initial_capital)

# --- 介面設定 ---
st.set_page_config(page_title="AI 股票戰艦 v17.0", layout="wide")
app_mode = st.sidebar.radio("切換模式", ["🔍 單股深度分析", "🚀 AI 自動巡航掃描"])

if app_mode == "🔍 單股深度分析":
    st.title("📊 智能股票深度分析 (支援 5分K 當沖版)")
    st.sidebar.header("1. 分析設定")
    sel = st.sidebar.selectbox("快速選擇股票", list(TW_STOCKS.keys()))
    ticker = st.sidebar.text_input("代碼", value="2330.TW") if sel == "✍️ 自訂輸入" else TW_STOCKS[sel]
    
    # 💡 新增：K線頻率選擇器
    interval_choice = st.sidebar.selectbox("K線頻率 (Timeframe)", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
    inv_map = {"日K (1d)": "1d", "15分K (15m)": "15m", "5分K (5m)": "5m"}
    interval_val = inv_map[interval_choice]

    p_map = {"3年": 1095, "1年": 365, "6個月": 180, "3個月": 90, "1個月": 30}
    p_sel = st.sidebar.selectbox("查詢期間", list(p_map.keys()), index=1)
    end_d = st.sidebar.date_input("結束日期", datetime.now())
    start_d = end_d - timedelta(days=p_map[p_sel])

    # 💡 防呆機制：分K最多只能查 60 天
    if interval_val in ["5m", "15m"]:
        if (end_d - start_d).days > 59:
            start_d = end_d - timedelta(days=59)
            st.sidebar.warning("⚠️ Yahoo API 限制：分K線最多僅支援近 60 天資料，已自動為您修正開始日期。")

    st.sidebar.markdown("---")
    strat = st.sidebar.selectbox("交易策略", STRATEGIES)
    tp, sl = st.sidebar.slider("停利 (%)", 5, 100, 20, 5), st.sidebar.slider("停損 (%)", 1, 50, 10, 1)

    if st.sidebar.button("開始深度分析"):
        with st.spinner('AI 分析中...'):
            cp, ch, cpct = get_latest_price(ticker)
            if cp: st.metric(f"⚡ {ticker.upper()} 最新報價", f"{cp:.2f}", f"{ch:.2f} ({cpct:.2f}%)")
            
            # 帶入 interval 參數
            df = fetch_stock_data(ticker, start_d, end_d, interval=interval_val)
            info = fetch_stock_info(ticker)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                df_bt = run_backtest(df, strat, sl/100, tp/100)
                
                latest_sig = df_bt['Signal'].iloc[-1]
                if latest_sig == 1: st.success(f"🤖 建議：買進 / 持有 ({strat})")
                else: st.warning(f"🤖 建議：賣出 / 觀望 ({strat})")

                tab1, tab2, tab3 = st.tabs([f"📈 技術分析 ({interval_choice})", "🏢 基本面與籌碼", "⏱️ 績效回測"])
                with tab1:
                    fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig_k.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(173, 204, 255, 0.2)'), showlegend=False), row=1, col=1)
                    fig_k.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(173, 204, 255, 0.2)'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)', name='布林通道'), row=1, col=1)
                    fig_k.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
                    
                    buys, sells = df_bt[df_bt['Action_Buy']], df_bt[df_bt['Action_Sell']]
                    fig_k.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=15), name='買入'), row=1, col=1)
                    fig_k.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='#FF4B4B', size=15), name='賣出'), row=1, col=1)
                    
                    vol_colors = ['#d62728' if row['Close'] < row['Open'] else '#2ca02c' for idx, row in df.iterrows()]
                    fig_k.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='成交量'), row=2, col=1)
                    
                    fig_k.update_layout(height=800, xaxis_rangeslider_visible=False, title=f"{ticker.upper()} 量價走勢圖 ({interval_choice})")
                    st.plotly_chart(fig_k, use_container_width=True)
                    
                    st.plotly_chart(go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB63FA'))]).update_layout(title="RSI", height=300, yaxis=dict(range=[0, 100])).add_hline(y=70).add_hline(y=30), use_container_width=True)
                    
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='快線'))
                    fig_m.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='慢線'))
                    fig_m.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='柱狀圖', marker_color=['#2ca02c' if v>=0 else '#d62728' for v in df['MACD_Hist']]))
                    st.plotly_chart(fig_m.update_layout(title="MACD", height=350), use_container_width=True)

                with tab2:
                    st.subheader("基本面概覽 (日頻數據)")
                    c1, c2, c3, c4 = st.columns(4)
                    pe, eps = info.get('trailingPE', 0), info.get('trailingEps', 0)
                    c1.metric("P/E", f"{pe:.2f}"); c2.metric("EPS", f"{eps:.2f}")
                    c3.metric("法人持股", f"{info.get('heldPercentInstitutions', 0)*100:.2f}%")
                    c4.metric("股息率", f"{info.get('dividendYield', 0)*100:.2f}%")
                    if pe > 0 and eps > 0:
                        mx, mn = info.get('fiftyTwoWeekHigh', df['Close'].max())/eps, info.get('fiftyTwoWeekLow', df['Close'].min())/eps
                        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=pe, title={'text': "P/E 歷史位階"}, gauge={'axis':{'range':[mn*0.8, mx*1.2]}, 'steps':[{'range':[0, mn+(mx-mn)*0.33], 'color':"lightgreen"}, {'range':[mn+(mx-mn)*0.66, 100], 'color':"salmon"}], 'threshold':{'line':{'color':"red", 'width':4}, 'value':pe}})).update_layout(height=280))

                with tab3:
                    fm, fs = df_bt['Market_Value'].iloc[-1], df_bt['Strategy_Value'].iloc[-1]
                    st.subheader("回測結果")
                    st.metric("策略最終價值", f"${fs:,.0f}", f"${fs-fm:,.0f} (vs 大盤/長期持有)")
                    st.plotly_chart(go.Figure().add_trace(go.Scatter(x=df_bt.index, y=df_bt['Market_Value'], name='大盤/長期持有', line=dict(dash='dot'))).add_trace(go.Scatter(x=df_bt.index, y=df_bt['Strategy_Value'], name='策略', line=dict(width=3))), use_container_width=True)

elif app_mode == "🚀 AI 自動巡航掃描":
    st.title("🚀 AI 自動選股雷達 (多週期掃描版)")
    st_autorefresh(interval=300000, key="fscancounter")
    st.sidebar.header("🎯 巡航設定")
    
    # 💡 雷達掃描也支援分K了
    scan_interval_choice = st.sidebar.selectbox("雷達掃描 K線頻率", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
    inv_map = {"日K (1d)": "1d", "15分K (15m)": "15m", "5分K (5m)": "5m"}
    scan_interval_val = inv_map[scan_interval_choice]

    selected_strats = st.sidebar.multiselect("策略多選", STRATEGIES, default=["纏論核心 (底背離+二買策略)"])
    is_auto = st.sidebar.toggle("啟用自動輪巡", value=True)
    categories = list(SCAN_POOLS.keys())
    current_cat = categories[st.session_state['scan_index']] if is_auto else st.sidebar.selectbox("指定板塊", categories)
    if is_auto: st.session_state['scan_index'] = (st.session_state['scan_index'] + 1) % len(categories)
    
    st.subheader(f"📡 目前掃描：【{current_cat}】 | 頻率：{scan_interval_choice}")
    buy_candidates = []
    end_d = datetime.now()
    # 💡 防呆機制：如果是分K，天數最多給 59 天
    start_d = end_d - timedelta(days=59 if scan_interval_val in ["5m", "15m"] else 180)
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    progress = st.progress(0); status = st.empty(); stocks = SCAN_POOLS[current_cat]
    
    for idx, (ticker, name) in enumerate(stocks.items()):
        status.text(f"掃描中: {name}...")
        try:
            time.sleep(0.15)
            # 帶入 interval 參數
            df = fetch_stock_data(ticker, start_d, end_d, interval=scan_interval_val)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                for strat in selected_strats:
                    df_sig = generate_signals(df, strat)
                    if df_sig['Signal'].iloc[-1] == 1:
                        price = round(df_sig['Close'].iloc[-1], 2)
                        res = {
                            "觸發時間": current_time_str, 
                            "頻率": scan_interval_choice, # 記錄觸發的 K線層級
                            "板塊": current_cat, "代碼": ticker, "股票名稱": name, 
                            "收盤價": price, "觸發策略": strat
                        }
                        buy_candidates.append(res)
                        save_record_to_csv(pd.DataFrame([res]))
        except: pass
        progress.progress((idx + 1) / len(stocks))
    
    st.session_state['scan_log'] = load_history()
    if buy_candidates:
        st.success(f"🔥 發現 {len(buy_candidates)} 個訊號！"); st.dataframe(pd.DataFrame(buy_candidates), use_container_width=True)
    
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