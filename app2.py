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

# --- 股票雷達池 ---
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

TW_STOCKS = {"✍️ 自訂輸入": ""}
TICKER_NAME_MAP = {}
for cat, stocks in SCAN_POOLS.items():
    for tkr, name in stocks.items():
        TW_STOCKS[f"{tkr} - {name}"] = tkr
        TICKER_NAME_MAP[tkr] = name
TW_STOCKS.update({"NVDA - 輝達": "NVDA", "TSLA - 特斯拉": "TSLA", "AAPL - 蘋果": "AAPL"})
TICKER_NAME_MAP.update({"NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果"})

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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

def get_company_name(ticker, info_dict):
    ticker_up = ticker.upper()
    if ticker_up in TICKER_NAME_MAP: return TICKER_NAME_MAP[ticker_up]
    return info_dict.get('shortName', '')

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker, start, end, interval="1d"):
    return yf.download(ticker, start=start, end=end, interval=interval, progress=False)

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'], df['Lower'] = df['MA20']+(df['std']*2), df['MA20']-(df['std']*2)
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
        low_20 = df_bt['Low'].rolling(window=20).min()
        hist_min_20 = df_bt['MACD_Hist'].rolling(window=20).min()
        div_buy = (df_bt['Low'] <= low_20) & (df_bt['MACD_Hist'] > hist_min_20.shift(1))
        
        last_low = df_bt['Low'].rolling(window=30).min().shift(5)
        sec_buy = (df_bt['Low'] > last_low) & (df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['MACD'].shift(1) < df_bt['Signal_Line'].shift(1))
        
        high_20 = df_bt['High'].rolling(window=20).max()
        hist_max_20 = df_bt['MACD_Hist'].rolling(window=20).max()
        div_sell = (df_bt['High'] >= high_20) & (df_bt['MACD_Hist'] < hist_max_20.shift(1))
        
        df_bt.loc[div_buy | sec_buy, 'Signal'] = 1
        df_bt.loc[div_sell | (df_bt['MACD'] < df_bt['Signal_Line']), 'Signal'] = 0
    elif strategy_choice == "纏論簡化版 (MACD 底背馳)":
        low_20 = df_bt['Low'].rolling(window=20).min()
        hist_min_20 = df_bt['MACD_Hist'].rolling(window=20).min()
        df_bt.loc[(df_bt['Low'] <= low_20) & (df_bt['MACD_Hist'] > hist_min_20.shift(1)), 'Signal'] = 1
        df_bt.loc[df_bt['MACD'] < df_bt['Signal_Line'], 'Signal'] = 0
    elif strategy_choice == "布林通道+RSI反轉":
        df_bt.loc[(df_bt['Low'] <= df_bt['Lower']) & (df_bt['RSI'] < 35), 'Signal'] = 1
        df_bt.loc[(df_bt['High'] >= df_bt['Upper']) | (df_bt['RSI'] > 70), 'Signal'] = 0
        
    df_bt['Signal'] = df_bt['Signal'].ffill().fillna(0)
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
st.set_page_config(page_title="AI 股票戰艦 v23.1", layout="wide")
app_mode = st.sidebar.radio("切換模式", ["🔍 單股深度分析", "🚀 AI 自動巡航掃描"])

if app_mode == "🔍 單股深度分析":
    st.title("📊 智能股票深度分析 (穩定預熱版)")
    st.sidebar.header("1. 分析設定")
    sel = st.sidebar.selectbox("快速選擇股票", list(TW_STOCKS.keys()))
    ticker = st.sidebar.text_input("輸入代碼", value="2330.TW") if sel == "✍️ 自訂輸入" else TW_STOCKS[sel]
    
    interval_choice = st.sidebar.selectbox("K線頻率", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
    inv_map = {"日K (1d)": "1d", "15分K (15m)": "15m", "5分K (5m)": "5m"}
    interval_val = inv_map[interval_choice]

    p_map = {"3年": 1095, "1年": 365, "6個月": 180, "3個月": 90, "1個月": 30, "20日": 20, "10日": 10, "5日": 5, "1日": 1, "✍️ 自訂": 0}
    default_index = 1 if interval_val == "1d" else 7 
    p_sel = st.sidebar.selectbox("查詢期間", list(p_map.keys()), index=default_index)
    
    # 💡 修復：統一讓時間產生純粹的 date 格式
    end_d = st.sidebar.date_input("結束日期 (預設今日)", datetime.now().date())
    
    if p_sel == "✍️ 自訂":
        start_d = st.sidebar.date_input("開始日期", end_d - timedelta(days=30))
    else:
        start_d = end_d - timedelta(days=p_map[p_sel])
        st.sidebar.text(f"📅 開始日期: {start_d.strftime('%Y-%m-%d')}")

    buffer_days = 60 if interval_val == "1d" else 15
    fetch_start = start_d - timedelta(days=buffer_days)

    if interval_val in ["5m", "15m"]:
        # 💡 修復：把包含時分秒的 datetime 轉換成純粹的 date
        min_allowed = (datetime.now() - timedelta(days=59)).date() 
        if fetch_start < min_allowed: fetch_start = min_allowed
        if start_d < min_allowed: 
            start_d = min_allowed
            st.sidebar.warning("⚠️ Yahoo 限制分K最多支援近 60 天，已自動修正日期。")

    st.sidebar.markdown("---")
    strat = st.sidebar.selectbox("交易策略", STRATEGIES)
    tp, sl = st.sidebar.slider("停利 (%)", 5, 100, 20, 5), st.sidebar.slider("停損 (%)", 1, 50, 10, 1)

    if st.sidebar.button("開始深度分析"):
        with st.spinner('載入資料與計算指標中...'):
            info = fetch_stock_info(ticker)
            cp, ch, cpct = get_latest_price(ticker)
            
            comp_name = get_company_name(ticker, info)
            display_name = f" - {comp_name}" if comp_name else ""
            
            if cp: st.metric(f"⚡ {ticker.upper()}{display_name} 最新報價", f"{cp:.2f}", f"{ch:.2f} ({cpct:.2f}%)")
            
            df = fetch_stock_data(ticker, fetch_start, end_d + timedelta(days=1), interval=interval_val)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                df_bt = run_backtest(df, strat, sl/100, tp/100)
                
                start_dt = pd.to_datetime(start_d)
                end_dt = pd.to_datetime(end_d) + timedelta(days=1)
                if df_bt.index.tz is not None:
                    start_dt = start_dt.tz_localize(df_bt.index.tz)
                    end_dt = end_dt.tz_localize(df_bt.index.tz)
                
                df_disp = df_bt[(df_bt.index >= start_dt) & (df_bt.index < end_dt)]
                
                if df_disp.empty:
                    st.error("⚠️ 在您選擇的日期內沒有交易資料 (可能是假日或剛好沒開盤)。")
                else:
                    sig_now = df_disp['Signal'].iloc[-1]
                    if sig_now == 1: st.success(f"🤖 目前建議：【買進/持有】(策略：{strat})")
                    else: st.warning(f"🤖 目前建議：【觀望/賣出】(策略：{strat})")

                    tab1, tab2, tab3 = st.tabs(["📈 技術分析 (量價與指標)", "🏢 基本面與籌碼", "⏱️ 績效回測報告"])
                    
                    with tab1:
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                        fig.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Upper'], line=dict(color='rgba(173, 204, 255, 0.2)'), showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Lower'], line=dict(color='rgba(173, 204, 255, 0.2)'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)', name='布林通道'), row=1, col=1)
                        fig.add_trace(go.Candlestick(x=df_disp.index, open=df_disp['Open'], high=df_disp['High'], low=df_disp['Low'], close=df_disp['Close'], name='K線'), row=1, col=1)
                        
                        buy_pts = df_disp[df_disp['Action_Buy']]; sell_pts = df_disp[df_disp['Action_Sell']]
                        fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=15), name='買入'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='#FF4B4B', size=15), name='賣出'), row=1, col=1)
                        
                        vol_colors = ['#d62728' if row['Close'] < row['Open'] else '#2ca02c' for idx, row in df_disp.iterrows()]
                        fig.add_trace(go.Bar(x=df_disp.index, y=df_disp['Volume'], marker_color=vol_colors, name='成交量'), row=2, col=1)
                        st.plotly_chart(fig.update_layout(height=650, xaxis_rangeslider_visible=False, title=f"{ticker.upper()} K線與量價走勢"), use_container_width=True)

                        st.plotly_chart(go.Figure(data=[go.Scatter(x=df_disp.index, y=df_disp['RSI'], line=dict(color='#AB63FA', width=2))]).update_layout(title="RSI 指標 (30/70)", height=300, yaxis=dict(range=[0, 100])).add_hline(y=70, line_dash="dash", line_color="red").add_hline(y=30, line_dash="dash", line_color="green"), use_container_width=True)
                        
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df_disp.index, y=df_disp['MACD'], name='快線', line=dict(color='#1f77b4')))
                        fig_macd.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Signal_Line'], name='慢線', line=dict(color='#ff7f0e')))
                        fig_macd.add_trace(go.Bar(x=df_disp.index, y=df_disp['MACD_Hist'], name='柱狀圖', marker_color=['#2ca02c' if v>=0 else '#d62728' for v in df_disp['MACD_Hist']]))
                        st.plotly_chart(fig_macd.update_layout(title="MACD 動能指標", height=350), use_container_width=True)

                    with tab2:
                        st.subheader("📊 基本面與籌碼概覽")
                        c1, c2, c3, c4 = st.columns(4)
                        pe = info.get('trailingPE', 0); eps = info.get('trailingEps', 0)
                        pb = info.get('priceToBook', 0); div = info.get('dividendYield', 0)
                        
                        c1.metric("P/E (本益比)", f"{pe:.2f}" if isinstance(pe, (int, float)) and pe > 0 else "N/A")
                        c2.metric("EPS (每股盈餘)", f"{eps:.2f}" if isinstance(eps, (int, float)) else "N/A")
                        c3.metric("法人持股比例", f"{info.get('heldPercentInstitutions', 0)*100:.2f}%")
                        c4.metric("殖利率", f"{div*100:.2f}%" if isinstance(div, float) else "N/A")
                        
                        st.markdown("---")
                        if isinstance(pe, (int, float)) and pe > 0 and isinstance(eps, (int, float)) and eps > 0:
                            max_pe = info.get('fiftyTwoWeekHigh', df['High'].max()) / eps
                            min_pe = info.get('fiftyTwoWeekLow', df['Low'].min()) / eps
                            fig_pe = go.Figure(go.Indicator(
                                mode="gauge+number", value=pe, title={'text': "P/E 歷史區間位階"},
                                gauge={'axis': {'range': [min_pe*0.8, max_pe*1.1]},
                                       'steps': [{'range': [0, min_pe+(max_pe-min_pe)*0.33], 'color': "lightgreen"},
                                                 {'range': [min_pe+(max_pe-min_pe)*0.66, 100], 'color': "salmon"}],
                                       'threshold': {'line': {'color': "red", 'width': 4}, 'value': pe}}
                            ))
                            st.plotly_chart(fig_pe.update_layout(height=300), use_container_width=True)
                        st.write("**📝 公司簡介:**", info.get('longBusinessSummary', '無提供相關資料。'))

                    with tab3:
                        fm, fs = df_disp['Market_Value'].iloc[-1], df_disp['Strategy_Value'].iloc[-1]
                        st.subheader("⏱️ 策略回測表現")
                        st.metric("策略最終資金", f"${fs:,.0f}", f"{fs-fm:,.0f} (超越大盤績效)")
                        fig_v = go.Figure()
                        fig_v.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Market_Value'], name='大盤/買進持有', line=dict(dash='dot', color='gray')))
                        fig_v.add_trace(go.Scatter(x=df_disp.index, y=df_disp['Strategy_Value'], name='策略執行績效', line=dict(width=3, color='blue')))
                        st.plotly_chart(fig_v.update_layout(height=450, title="資金成長曲線對比"), use_container_width=True)
            else:
                st.error("⚠️ 無法獲取資料，請檢查代碼是否正確。")

elif app_mode == "🚀 AI 自動巡航掃描":
    st.title("🚀 AI 自動選股雷達 (多週期掃描版)")
    st_autorefresh(interval=300000, key="fscancounter")
    st.sidebar.header("🎯 巡航設定")
    
    scan_interval_choice = st.sidebar.selectbox("雷達頻率", ["日K (1d)", "15分K (15m)", "5分K (5m)"])
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
    start_d = end_d - timedelta(days=59 if scan_interval_val != "1d" else 180)
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    progress = st.progress(0); status = st.empty(); stocks = SCAN_POOLS[current_cat]
    
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
                    if df_sig['Signal'].iloc[-1] == 1:
                        price = round(df_sig['Close'].iloc[-1], 2)
                        res = {"觸發時間": current_time_str, "頻率": scan_interval_choice, "板塊": current_cat, "代碼": ticker, "股票名稱": name, "收盤價": price, "觸發策略": strat}
                        buy_candidates.append(res); save_record_to_csv(pd.DataFrame([res]))
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