import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# --- 0. 初始化系統記憶體 ---
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

# 熱門選單
TW_STOCKS = {
    "✍️ 自訂輸入": "",
    "2330.TW - 台積電": "2330.TW",
    "2317.TW - 鴻海": "2317.TW",
    "2454.TW - 聯發科": "2454.TW",
    "NVDA - 輝達": "NVDA",
    "TSLA - 特斯拉": "TSLA",
    "AAPL - 蘋果": "AAPL",
    "0050.TW - 元大台灣50": "0050.TW"
}

# --- 核心函數：報價、資料、計算 ---
def get_latest_price(ticker_str):
    try:
        tkr = yf.Ticker(ticker_str)
        fast = tkr.fast_info
        return fast.last_price, (fast.last_price - fast.previous_close), ((fast.last_price - fast.previous_close)/fast.previous_close*100)
    except: return None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

def calculate_indicators(df):
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

def run_backtest_with_sltp(df, strategy_choice, sl_pct, tp_pct, initial_capital=100000):
    df_bt = df.copy()
    df_bt['Signal'] = 0
    if strategy_choice == "均線黃金交叉 (20MA & 50MA)":
        df_bt['SMA_20'] = df_bt['Close'].rolling(window=20).mean()
        df_bt['SMA_50'] = df_bt['Close'].rolling(window=50).mean()
        df_bt.loc[df_bt['SMA_20'] > df_bt['SMA_50'], 'Signal'] = 1
    elif strategy_choice == "RSI 超買超賣 (30買/70賣)":
        df_bt['Signal'] = np.nan
        df_bt.loc[df_bt['RSI'] < 30, 'Signal'] = 1  
        df_bt.loc[df_bt['RSI'] > 70, 'Signal'] = 0  
        df_bt['Signal'] = df_bt['Signal'].ffill().fillna(0)
    elif strategy_choice == "MACD 黃金交叉/死亡交叉":
        df_bt.loc[df_bt['MACD'] > df_bt['Signal_Line'], 'Signal'] = 1

    df_bt['Position'] = 0
    df_bt['Action_Buy'] = False
    df_bt['Action_Sell'] = False
    pos, entry = 0, 0.0
    
    for i in range(1, len(df_bt)):
        curr = df_bt['Close'].iloc[i]
        sig = df_bt['Signal'].iloc[i-1]
        if pos == 1:
            if curr <= entry * (1-sl_pct) or curr >= entry * (1+tp_pct) or sig == 0:
                pos = 0
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
        elif pos == 0 and sig == 1:
            pos, entry = 1, curr
            df_bt.iloc[i, df_bt.columns.get_loc('Action_Buy')] = True
        df_bt.iloc[i, df_bt.columns.get_loc('Position')] = pos

    df_bt['Market_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1).fillna(0) * df_bt['Market_Return']
    df_bt['Market_Value'] = initial_capital * (1 + df_bt['Market_Return']).cumprod()
    df_bt['Strategy_Value'] = initial_capital * (1 + df_bt['Strategy_Return']).cumprod()
    return df_bt.fillna(initial_capital)

# --- 2. 介面設定 ---
st.set_page_config(page_title="AI 股票分析系統", layout="wide")
st.title("📊 智能股票分析系統 v7.1")

# 側邊欄
st.sidebar.header("1. 設定")
sel = st.sidebar.selectbox("快速選擇股票", list(TW_STOCKS.keys()))
ticker = st.sidebar.text_input("代碼", value="2330.TW") if sel == "✍️ 自訂輸入" else TW_STOCKS[sel]

st.sidebar.markdown("---")
st.sidebar.header("2. 時間區間")
p_map = {"3年": 1095, "1年": 365, "6個月": 180, "3個月": 90, "1個月": 30}
p_sel = st.sidebar.selectbox("查詢期間", list(p_map.keys()), index=1)
end_d = st.sidebar.date_input("結束日期", datetime.now())
start_d = end_d - timedelta(days=p_map[p_sel])

st.sidebar.markdown("---")
st.sidebar.header("3. 策略/風控")
strat = st.sidebar.selectbox("交易策略", ["均線黃金交叉 (20MA & 50MA)", "RSI 超買超賣 (30買/70賣)", "MACD 黃金交叉/死亡交叉"])
tp = st.sidebar.slider("停利目標 (%)", 5, 100, 20, 5)
sl = st.sidebar.slider("停損底線 (%)", 1, 50, 10, 1)

if st.sidebar.button("開始分析"):
    t_up = ticker.upper()
    if t_up not in st.session_state['search_history']:
        st.session_state['search_history'].insert(0, t_up)
        if len(st.session_state['search_history']) > 5: st.session_state['search_history'].pop()
    
    with st.spinner('運算中...'):
        cp, ch, cpct = get_latest_price(ticker)
        if cp:
            c1, c2 = st.columns([1, 3])
            c1.metric(f"⚡ {t_up} 最新報價", f"{cp:.2f}", f"{ch:.2f} ({cpct:.2f}%)")
            st.markdown("---")

        df = fetch_stock_data(ticker, start_d, end_d)
        info = fetch_stock_info(ticker)
        
        if df.empty:
            st.error("找不到數據。")
        else:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df = calculate_indicators(df)
            df_bt = run_backtest_with_sltp(df, strat, sl/100, tp/100)
            
            latest_sig = df_bt['Signal'].iloc[-1]
            st.markdown(f"### 🤖 系統最新操作建議 ({t_up})")
            if latest_sig == 1: st.success(f"**【建議：買進 / 持有】** 根據 {strat}。")
            else: st.warning(f"**【建議：賣出 / 觀望】** 根據 {strat}。")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 技術分析", "🏢 財報基本面", "⏱️ 回測報告", "📖 教學說明"])
            
            with tab1:
                fig_k = go.Figure(data=[go.Candlestick(x=df_bt.index, open=df_bt['Open'], high=df_bt['High'], low=df_bt['Low'], close=df_bt['Close'], name='K線')])
                buys = df_bt[df_bt['Action_Buy']]
                sells = df_bt[df_bt['Action_Sell']]
                fig_k.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=14), name='買進'))
                fig_k.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='#FF0000', size=14), name='賣出'))
                fig_k.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
                st.plotly_chart(fig_k, use_container_width=True)
                
                fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB63FA', width=2))])
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI 指標", height=350, yaxis=dict(range=[0, 100]), margin=dict(t=50, b=10))
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='快線', line=dict(color='#1f77b4')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='慢線', line=dict(color='#ff7f0e')))
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='柱狀圖', marker_color=['#2ca02c' if v>=0 else '#d62728' for v in df['MACD_Hist']]))
                fig_macd.update_layout(title="MACD 指標", height=400, margin=dict(t=50, b=10))
                st.plotly_chart(fig_macd, use_container_width=True)

            with tab2:
                st.subheader("基本面數據")
                c1, c2, c3, c4 = st.columns(4)
                pe_ratio = info.get('trailingPE', 'N/A')
                eps = info.get('trailingEps', 'N/A')
                pb = info.get('priceToBook', 'N/A')
                div = info.get('dividendYield', 'N/A')
                forward_pe = info.get('forwardPE', 'N/A')
                
                c1.metric("當前 P/E (本益比)", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)
                c2.metric("EPS (每股盈餘)", f"{eps:.2f}" if isinstance(eps, (int, float)) else eps)
                c3.metric("P/B (股價淨值比)", f"{pb:.2f}" if isinstance(pb, (int, float)) else pb)
                c4.metric("股息殖利率", f"{div*100:.2f}%" if isinstance(div, (float, int)) else div)
                
                st.markdown("---")
                st.subheader("⚖️ 本益比 (P/E) 估值評估")
                if isinstance(pe_ratio, (int, float)) and isinstance(eps, (int, float)) and eps > 0:
                    high_52w = info.get('fiftyTwoWeekHigh', df['High'].max())
                    low_52w = info.get('fiftyTwoWeekLow', df['Low'].min())
                    max_pe, min_pe = high_52w / eps, low_52w / eps
                    range_diff = max_pe - min_pe
                    
                    fig_pe = go.Figure(go.Indicator(
                        mode = "gauge+number", value = pe_ratio, domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "當前本益比區間 (過去1年)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [max(0, min_pe * 0.8), max_pe * 1.1]}, 'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, min_pe + range_diff * 0.33], 'color': "lightgreen"},
                                {'range': [min_pe + range_diff * 0.33, min_pe + range_diff * 0.66], 'color': "#FFD700"},
                                {'range': [min_pe + range_diff * 0.66, max_pe * 1.5], 'color': "salmon"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pe_ratio}
                        }
                    ))
                    fig_pe.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_pe, use_container_width=True)
                    st.info(f"**估值參考說明：**\n* 依據該公司過去 52 週的股價波動，過去一年的本益比約落在 **{min_pe:.1f} 倍** (最便宜) 到 **{max_pe:.1f} 倍** (最昂貴) 之間。\n* 預估未來的本益比 (Forward P/E) 為：**{forward_pe if isinstance(forward_pe, str) else round(forward_pe, 2)}** (若低於當前 P/E，代表市場預期未來獲利會成長)。")
                else:
                    st.warning("⚠️ 公司目前 EPS 為負值（虧損），或缺乏完整財報數據，無法進行本益比估值比較。")
                
                st.markdown("---")
                st.write("**公司簡介:**", info.get('longBusinessSummary', '無資料'))

            with tab3:
                final_m = df_bt['Market_Value'].iloc[-1]
                final_s = df_bt['Strategy_Value'].iloc[-1]
                diff = final_s - final_m
                
                st.subheader("策略回測表現")
                m1, m2, m3 = st.columns(3)
                m1.metric("大盤持有最終價值", f"${final_m:,.0f}")
                m2.metric("策略最終價值", f"${final_s:,.0f}", f"${diff:,.0f} (vs 大盤)")
                m3.metric("總進場次數", f"{len(buys)} 次")

                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Market_Value'], name='大盤持有', line=dict(color='gray', dash='dot')))
                fig_v.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Strategy_Value'], name='策略資金曲線', line=dict(color='#EF553B', width=3)))
                fig_v.update_layout(title="資金成長對比", height=500)
                st.plotly_chart(fig_v, use_container_width=True)

            with tab4:
                # --- 💡 完整還原並升級教學說明 ---
                st.header("📖 技術指標與基本面教學說明")
                
                st.subheader("1️⃣ RSI 相對強弱指標 (Relative Strength Index)")
                st.markdown("""
                * **碰到 70 上方的紅虛線：** 代表現在大家都在買，價格可能太高了，進入「超買區」，要小心回檔下跌風險。
                * **碰到 30 下方的綠虛線：** 代表大家都在賣，價格可能被低估，進入「超賣區」，是潛在的反彈買點。
                """)
                st.markdown("---")
                
                st.subheader("2️⃣ MACD 平滑異同移動平均線")
                st.markdown("""
                * **MACD 快線（通常是藍線）：** 代表短期的價格動能。計算公式是 `12日指數移動平均(EMA) - 26日指數移動平均(EMA)`。
                * **Signal 慢線（通常是橘線）：** 代表長期的價格趨勢。它是 MACD 快線的 `9日指數移動平均`，用來平滑快線的波動，作為判斷標準。
                * **柱狀圖 (Histogram, MACD - Signal)：** 快線減去慢線的差值。柱狀圖在 0 軸以上（綠柱）代表多頭動能強，0 軸以下（紅柱）代表空頭動能強。
                """)
                st.markdown("#### 💡 最經典的 MACD 交易策略：交叉訊號")
                st.write("多數投資人會利用快線與慢線的交叉來尋找買賣點：")
                st.markdown("""
                * ✅ **黃金交叉（買進訊號）：** MACD 快線由下往上穿過 Signal 慢線，且柱狀圖由負轉正。這代表短期動能轉強，股價可能準備發動一波漲勢。
                * 🔻 **死亡交叉（賣出訊號）：** MACD 快線由上往下穿過 Signal 慢線，且柱狀圖由正轉負。這代表短期動能轉弱，股價可能面臨回檔或下跌。
                """)
                st.markdown("---")
                
                st.subheader("3️⃣ EPS 與 本益比 (P/E Ratio) 基礎概念")
                st.markdown("""
                * **EPS (每股盈餘)：** 稅後淨利 ÷ 發行總股數。代表公司今年為每一股賺了多少錢。好的 EPS 應該要大於 0，且最好能「年年成長」。
                * **P/E (本益比)：** 股價 ÷ EPS。代表你買進這檔股票，需要幾年才能回本。
                    * **低本益比 (<10~12)：** 股價可能被低估，相對便宜。
                    * **合理區間 (12~18)：** 多數穩定獲利公司的常態。
                    * **高本益比 (>20~30)：** 股價偏貴，但通常是因為市場看好它未來的爆發性成長（例如 AI 概念股）。
                """)

st.sidebar.markdown("---")
if st.session_state['search_history']:
    st.sidebar.write("🕒 最近查詢：", ", ".join(st.session_state['search_history']))
