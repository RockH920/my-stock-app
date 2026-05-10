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
    # 布林通道 (20MA, 2 Std)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['std'] * 2)
    df['Lower'] = df['MA20'] - (df['std'] * 2)
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

    elif strategy_choice == "纏論核心 (底背離+二買策略)":
        # 1. 偵測底背離 (一買)
        low_20 = df_bt['Low'].rolling(window=20).min()
        hist_min_20 = df_bt['MACD_Hist'].rolling(window=20).min()
        divergence_buy = (df_bt['Low'] <= low_20) & (df_bt['MACD_Hist'] > hist_min_20.shift(1))
        
        # 2. 偵測二買 (回踩不破底且 MACD 金叉)
        last_low = df_bt['Low'].rolling(window=30).min().shift(5)
        second_buy = (df_bt['Low'] > last_low) & (df_bt['MACD'] > df_bt['Signal_Line']) & (df_bt['MACD'].shift(1) < df_bt['Signal_Line'].shift(1))
        
        df_bt.loc[divergence_buy | second_buy, 'Signal'] = 1
        
        # 3. 偵測頂背離 (一賣)
        high_20 = df_bt['High'].rolling(window=20).max()
        hist_max_20 = df_bt['MACD_Hist'].rolling(window=20).max()
        divergence_sell = (df_bt['High'] >= high_20) & (df_bt['MACD_Hist'] < hist_max_20.shift(1))
        
        df_bt.loc[divergence_sell | (df_bt['MACD'] < df_bt['Signal_Line']), 'Signal'] = 0
        df_bt['Signal'] = df_bt['Signal'].replace(0, np.nan).ffill().fillna(0)

    elif strategy_choice == "纏論簡化版 (MACD 底背馳)":
        # 💡 邏輯：價格創 20 日新低，但 MACD 柱狀圖比上次創低時高
        df_bt['Lowest_20'] = df_bt['Low'].rolling(window=20).min()
        df_bt['Hist_Min_20'] = df_bt['MACD_Hist'].rolling(window=20).min()
        # 底背馳條件：破底但力道衰竭
        buy_cond = (df_bt['Low'] <= df_bt['Lowest_20']) & (df_bt['MACD_Hist'] > df_bt['Hist_Min_20'].shift(1))
        df_bt.loc[buy_cond, 'Signal'] = 1
        # 出現買訊後持續持有，直到 MACD 快慢線死叉或觸發風控
        df_bt.loc[df_bt['MACD'] < df_bt['Signal_Line'], 'Signal'] = 0
        df_bt['Signal'] = df_bt['Signal'].replace(0, np.nan).ffill().fillna(0)

    elif strategy_choice == "布林通道+RSI反轉":
        # 觸碰下軌且 RSI < 35 為買入
        df_bt.loc[(df_bt['Low'] <= df_bt['Lower']) & (df_bt['RSI'] < 35), 'Signal'] = 1
        # 觸碰上軌且 RSI > 65 為賣出
        df_bt.loc[(df_bt['High'] >= df_bt['Upper']) | (df_bt['RSI'] > 70), 'Signal'] = 0
        df_bt['Signal'] = df_bt['Signal'].replace(0, np.nan).ffill().fillna(0)

    df_bt['Position'] = 0
    df_bt['Action_Buy'] = False
    df_bt['Action_Sell'] = False
    pos, entry = 0, 0.0
    for i in range(1, len(df_bt)):
        curr = df_bt['Close'].iloc[i]
        sig = df_bt['Signal'].iloc[i-1]
        if pos == 1:
            if curr <= entry*(1-sl_pct) or curr >= entry*(1+tp_pct) or sig == 0:
                pos = 0
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
        elif pos == 0 and sig == 1:
            pos, entry = 1, curr
            df_bt.iloc[i, df_bt.columns.get_loc('Action_Buy')] = True
        df_bt.iloc[i, df_bt.columns.get_loc('Position')] = pos

    df_bt['Strategy_Return'] = df_bt['Position'].shift(1).fillna(0) * df_bt['Close'].pct_change()
    df_bt['Market_Value'] = initial_capital * (1 + df_bt['Close'].pct_change()).cumprod()
    df_bt['Strategy_Value'] = initial_capital * (1 + df_bt['Strategy_Return']).cumprod()
    return df_bt.fillna(initial_capital)
    
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
st.title("📊 智能股票分析系統 v10.0 by Rock)")

# 側邊欄
st.sidebar.header("1. 設定")
sel = st.sidebar.selectbox("快速選擇股票", list(TW_STOCKS.keys()))
ticker = st.sidebar.text_input("代碼", value="2330.TW") if sel == "✍️ 自訂輸入" else TW_STOCKS[sel]

# 時間區間
st.sidebar.markdown("---")
st.sidebar.header("2. 時間區間")
period_map = {
    "3年": 365*3,
    "1年": 365,
    "6個月": 30*6,
    "3個月": 30*3,
    "1個月": 30,
    "自訂": None
}
selected_period = st.sidebar.selectbox("選擇查詢期間", list(period_map.keys()), index=1) 

end_d = st.sidebar.date_input("結束日期", datetime.now())

if selected_period != "自訂":
    days = period_map[selected_period]
    start_d = end_d - timedelta(days=days)
    st.sidebar.info(f"已自動設定開始日期：\n{start_d.strftime('%Y-%m-%d')}")
else:
    start_d = st.sidebar.date_input("開始日期", datetime.now() - timedelta(days=365))

st.sidebar.markdown("---")
st.sidebar.header("3. 策略/風控")
strat = st.sidebar.selectbox("交易策略", ["均線黃金交叉 (20MA & 50MA)", "RSI 超買超賣 (30買/70賣)", "MACD 黃金交叉/死亡交叉","纏論核心 (底背離+二買策略)",
"纏論簡化版 (MACD 底背馳)","布林通道+RSI反轉"])
tp = st.sidebar.slider("停利目標 (%)", 5, 100, 20, 5)
sl = st.sidebar.slider("停損底線 (%)", 1, 50, 10, 1)

if st.sidebar.button("開始分析"):
    t_up = ticker.upper()
    if t_up not in st.session_state['search_history']:
        st.session_state['search_history'].insert(0, t_up)
        if len(st.session_state['search_history']) > 5: st.session_state['search_history'].pop()
    
    with st.spinner('運算中...'):
        # ⚡ 即時報價
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
            
            # 操作建議
            latest_sig = df_bt['Signal'].iloc[-1]
            st.markdown(f"### 🤖 系統最新操作建議 ({t_up})")
            if latest_sig == 1: st.success(f"**【建議：買進 / 持有】** 根據 {strat}。")
            else: st.warning(f"**【建議：賣出 / 觀望】** 根據 {strat}。")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 技術分析", "🏢 財報基本面", "⏱️ 回測報告", "📖 教學"])
            
            with tab1:
                # K線圖 + 布林通道
                fig_k = go.Figure()
                # 畫布林帶陰影
                fig_k.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(173, 204, 255, 0.2)'), showlegend=False))
                fig_k.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(173, 204, 255, 0.2)'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)', name='布林通道'))
                # 畫K線
                fig_k.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                # 標記買賣點
                buys, sells = df_bt[df_bt['Action_Buy']], df_bt[df_bt['Action_Sell']]
                fig_k.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='#00FF00', size=15), name='買入'))
                fig_k.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='#FF4B4B', size=15), name='賣出'))
                fig_k.update_layout(height=650, xaxis_rangeslider_visible=False, title=f"{t_up} 布林通道 K 線圖")
                st.plotly_chart(fig_k, use_container_width=True)
                
                
                # 💡 優化：RSI 高度增加，增加邊界線
                fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#AB63FA', width=2))])
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI 指標 (上下空間已加大)", height=350, yaxis=dict(range=[0, 100]), margin=dict(t=50, b=10))
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # 💡 優化：MACD 高度增加
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='快線', line=dict(color='#1f77b4')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='慢線', line=dict(color='#ff7f0e')))
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='柱狀圖', marker_color=['#2ca02c' if v>=0 else '#d62728' for v in df['MACD_Hist']]))
                fig_macd.update_layout(title="MACD 指標 (趨勢觀察更清晰)", height=400, margin=dict(t=50, b=10))
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
                # 籌碼面 (簡易法人持股比例)
                st.subheader("🕵️ 籌碼與股權概覽")
                col_c1, col_c2, col_c3 = st.columns(3)
                ins_ratio = info.get('heldPercentInstitutions', 0)
                col_c1.metric("法人持股比例", f"{ins_ratio*100:.2f}%")
                col_c2.metric("空單餘額 (Short Ratio)", f"{info.get('shortRatio', 0):.2f}")
                col_c3.metric("內部人持股", f"{info.get('heldPercentInsiders', 0)*100:.2f}%")
                
                                
                # 確保 PE 與 EPS 都有數字，且 EPS 大於 0 才能算估值
                if isinstance(pe_ratio, (int, float)) and isinstance(eps, (int, float)) and eps > 0:
                    # 抓取 52週最高與最低價 (如果 API 沒給，就從歷史數據裡抓)
                    high_52w = info.get('fiftyTwoWeekHigh', df['High'].max())
                    low_52w = info.get('fiftyTwoWeekLow', df['Low'].min())
                    
                    # 計算過去一年的本益比極值
                    max_pe = high_52w / eps
                    min_pe = low_52w / eps
                    
                    # 切割三個區間：便宜(綠)、合理(黃)、昂貴(紅)
                    range_diff = max_pe - min_pe
                    
                    fig_pe = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = pe_ratio,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "當前本益比區間 (過去1年)", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [max(0, min_pe * 0.8), max_pe * 1.1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, min_pe + range_diff * 0.33], 'color': "lightgreen"},
                                {'range': [min_pe + range_diff * 0.33, min_pe + range_diff * 0.66], 'color': "#FFD700"}, # 金黃色
                                {'range': [min_pe + range_diff * 0.66, max_pe * 1.5], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': pe_ratio
                            }
                        }
                    ))
                    fig_pe.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_pe, use_container_width=True)
                    
                    st.info(f"""
                    **估值參考說明：**
                    * 依據該公司過去 52 週的股價波動，過去一年的本益比約落在 **{min_pe:.1f} 倍** (最便宜) 到 **{max_pe:.1f} 倍** (最昂貴) 之間。
                    * 預估未來的本益比 (Forward P/E) 為：**{forward_pe:.2f}** (若低於當前 P/E，代表市場預期未來獲利會成長)。
                    """)
                else:
                    st.warning("⚠️ 公司目前 EPS 為負值（虧損），或缺乏完整財報數據，無法進行本益比估值比較。")
                
                st.markdown("---")
                st.write("**公司簡介:**", info.get('longBusinessSummary', '無資料'))

            with tab3:
                # 💡 優化：把大盤差價找回來
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
                st.header("📖 五大策略教學")
                st.write("本系統整合了以下交易邏輯，你可以切換測試：")
                st.markdown("""
                1. **均線黃金交叉**: 趨勢跟隨，20MA 穿過 50MA 時買入，適合大波段。
                2. **RSI 超買超賣**: 逆勢策略，RSI < 30 買入，> 70 賣出。
                3. **MACD 交叉**: 經典指標，快線穿過慢線時買入。
                4. **纏論核心 (背離+二買)**: 觀察價格創新低但動能未創新低的「底背離」現象。
                5. **布林+RSI 反轉**: 雙重確認，當價格觸碰布林下軌且 RSI 呈現超賣時入場。
                6. **布林通道 (Bollinger Bands)**:由 20MA 加上上下各 2 個標準差組成。股價碰觸下軌通常視為超賣。

                📖 纏論核心速成教學
                1. **底背離 (一買)**
                當價格創下新低，但 MACD 能量柱卻比前一次低點還要短，代表「雖然在跌，但賣壓枯竭」。
                2. **二類買點 (二買)**
                一買後的反彈拉回，低點不破前低，且 MACD 再次金叉。這是「趨勢確認」的安全入場點。
                3. **頂背離 (一賣)**
                當價格創新高，但 MACD 動能卻衰退，是強烈的見頂訊號。""")

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