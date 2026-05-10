import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# --- 1. 核心計算函數 ---
def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

def run_backtest_with_sltp(df, strategy_choice, sl_pct, tp_pct, initial_capital=100000):
    df_bt = df.copy()
    
    # 產生原始策略訊號
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

    # 逐日模擬交易 (加入停損停利)
    df_bt['Position'] = 0
    df_bt['Action_Buy'] = False
    df_bt['Action_Sell'] = False
    
    position = 0      
    entry_price = 0.0 
    
    for i in range(1, len(df_bt)):
        current_price = df_bt['Close'].iloc[i]
        prev_signal = df_bt['Signal'].iloc[i-1] 
        
        if position == 1:
            if current_price <= entry_price * (1 - sl_pct):
                position = 0
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
            elif current_price >= entry_price * (1 + tp_pct):
                position = 0
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
            elif prev_signal == 0:
                position = 0
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Sell')] = True
                
        elif position == 0:
            if prev_signal == 1:
                position = 1
                entry_price = current_price
                df_bt.iloc[i, df_bt.columns.get_loc('Action_Buy')] = True
                
        df_bt.iloc[i, df_bt.columns.get_loc('Position')] = position

    # 計算績效
    df_bt['Market_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1).fillna(0) * df_bt['Market_Return']
    df_bt['Market_Value'] = initial_capital * (1 + df_bt['Market_Return']).cumprod()
    df_bt['Strategy_Value'] = initial_capital * (1 + df_bt['Strategy_Return']).cumprod()
    df_bt['Market_Value'] = df_bt['Market_Value'].fillna(initial_capital)
    df_bt['Strategy_Value'] = df_bt['Strategy_Value'].fillna(initial_capital)
    
    return df_bt

# --- 2. 系統介面與參數設定 ---
st.set_page_config(page_title="AI 股票分析系統", layout="wide")
st.title("📊 智能股票分析系統 v6.3")

st.sidebar.header("1. 基礎設定")
ticker = st.sidebar.text_input("輸入股票代碼", value="NVDA") 
start_date = st.sidebar.date_input("開始日期", datetime.now() - timedelta(days=365*2)) 
end_date = st.sidebar.date_input("結束日期", datetime.now())

st.sidebar.markdown("---")
st.sidebar.header("2. 策略與風險控管")
strategy_choice = st.sidebar.selectbox(
    "選擇交易策略", 
    ["均線黃金交叉 (20MA & 50MA)", "RSI 超買超賣 (30買/70賣)", "MACD 黃金交叉/死亡交叉"]
)

tp_input = st.sidebar.slider("停利目標 (%)", min_value=5, max_value=100, value=20, step=5)
sl_input = st.sidebar.slider("停損底線 (%)", min_value=1, max_value=50, value=10, step=1)

if st.sidebar.button("開始分析"):
    with st.spinner('正在獲取數據與進行運算...'):
        df = yf.download(ticker, start=start_date, end=end_date)
        stock_info = yf.Ticker(ticker).info
        
        if df.empty:
            st.error("找不到數據，請確認代碼是否正確。")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df = calculate_indicators(df)
            sl_pct = sl_input / 100.0
            tp_pct = tp_input / 100.0
            df_bt = run_backtest_with_sltp(df, strategy_choice, sl_pct, tp_pct)
            
            # --- 最新狀態建議面板 ---
            latest_signal = df_bt['Signal'].iloc[-1]
            last_date = df_bt.index[-1].strftime("%Y-%m-%d")
            
            st.markdown("### 🤖 系統最新操作建議")
            if latest_signal == 1:
                st.success(f"**【建議：買進 / 持有】** 根據 {strategy_choice}，目前指標偏多。(數據日期: {last_date})")
            else:
                st.warning(f"**【建議：賣出 / 觀望】** 根據 {strategy_choice}，目前指標偏空，建議空手。(數據日期: {last_date})")
            
            st.markdown("---")
            
            # --- 新增第四個分頁：教學說明 ---
            tab1, tab2, tab3, tab4 = st.tabs(["📈 技術分析與訊號", "🏢 財報基本面", "⏱️ 風險控管回測報告", "📖 指標教學說明"])
            
            # 分頁 1: K線圖 + RSI + MACD
            with tab1:
                fig_price = go.Figure(data=[go.Candlestick(x=df_bt.index,
                                open=df_bt['Open'], high=df_bt['High'],
                                low=df_bt['Low'], close=df_bt['Close'], name='K線')])
                
                buy_dates = df_bt[df_bt['Action_Buy']]
                sell_dates = df_bt[df_bt['Action_Sell']]
                
                fig_price.add_trace(go.Scatter(x=buy_dates.index, y=buy_dates['Low'] * 0.95,
                    mode='markers', marker=dict(symbol='triangle-up', color='green', size=15), name='買進'))
                fig_price.add_trace(go.Scatter(x=sell_dates.index, y=sell_dates['High'] * 1.05,
                    mode='markers', marker=dict(symbol='triangle-down', color='red', size=15), name='賣出'))
                fig_price.update_layout(title=f"股價走勢與買賣點 (停損:{sl_input}% / 停利:{tp_input}%)", height=450, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_price, use_container_width=True)
                
                fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple'))])
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI 相對強弱指標", height=200, yaxis=dict(range=[0, 100]), margin=dict(t=30, b=10))
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD 快線', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal 慢線', line=dict(color='orange')))
                colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱狀', marker_color=colors))
                fig_macd.update_layout(title="MACD 趨勢指標", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_macd, use_container_width=True)

            # 分頁 2: 財報基本面
            with tab2:
                st.subheader(f"{stock_info.get('shortName', ticker)} 基本面數據")
                col1, col2, col3, col4 = st.columns(4)
                pe_ratio = stock_info.get('trailingPE', 'N/A')
                eps = stock_info.get('trailingEps', 'N/A')
                pb_ratio = stock_info.get('priceToBook', 'N/A')
                dividend = stock_info.get('dividendYield', 'N/A')
                
                col1.metric("本益比 (P/E)", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)
                col2.metric("每股盈餘 (EPS)", f"{eps:.2f}" if isinstance(eps, float) else eps)
                col3.metric("股價淨值比 (P/B)", f"{pb_ratio:.2f}" if isinstance(pb_ratio, float) else pb_ratio)
                if isinstance(dividend, float): col4.metric("股息殖利率", f"{dividend*100:.2f}%")
                else: col4.metric("股息殖利率", dividend)
                
                st.markdown("---")
                st.write("**公司簡介:**")
                st.write(stock_info.get('longBusinessSummary', '無公司簡介資料。'))

            # 分頁 3: 回測報告
            with tab3:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Market_Value'], name='大盤無腦持有', line=dict(color='gray')))
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Strategy_Value'], name='策略資金曲線', line=dict(color='red', width=2)))
                fig_bt.update_layout(title="資金成長對比圖", yaxis_title="資金餘額", height=450)
                st.plotly_chart(fig_bt, use_container_width=True)
                
                trade_count = len(buy_dates)
                final_market = df_bt['Market_Value'].dropna().iloc[-1]
                final_strategy = df_bt['Strategy_Value'].dropna().iloc[-1]
                
                st.markdown("### 📊 績效統計")
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("無腦持有最終結算", f"${final_market:,.0f}")
                col_b2.metric("策略最終結算", f"${final_strategy:,.0f}", f"${(final_strategy - final_market):,.0f} (vs 大盤)")
                col_b3.metric("總進場次數", f"{trade_count} 次")
            
            # 分頁 4: 教學說明
            with tab4:
                st.header("📖 技術指標教學說明")
                
                st.subheader("1️⃣ RSI 相對強弱指標")
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
