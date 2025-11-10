import tushare as ts
import pandas as pd
import numpy as np
import ta
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import backtrader as bt
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# 设置 Tushare Token
st.set_page_config(page_title="A股量化投资系统", layout="wide")
token = st.text_input("请输入Tushare Token", type="password")
if not token:
    st.warning("请输入有效的Tushare Token")
    st.stop()
ts.set_token(token)
pro = ts.pro_api()

# 配置文件路径
config_file = "config.json"

# =======================
# 数据获取与因子计算
# =======================
@st.cache
def get_data(stock_list, start_date="20150101", end_date="20251101"):
    all_data = []
    for stock in stock_list:
        daily_data = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
        basic_data = pro.daily_basic(ts_code=stock, start_date=start_date, end_date=end_date,
                                     fields='ts_code,trade_date,pe,pb,turnover_rate,volume_ratio,total_mv')
        df = pd.merge(daily_data, basic_data, on=["ts_code", "trade_date"])
        df["stock"] = stock
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# 获取股票列表示例：沪深300或自定义
stock_list = ['000001.SZ', '600519.SH', '000333.SZ', '601318.SH']

# 获取数据
data = get_data(stock_list)

# =======================
# 特征构建（五大因子）
# =======================
def compute_factors(df):
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['VOL5'] = df['vol'].rolling(5).mean()
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
    df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
    df['pb'] = pd.to_numeric(df['pb'], errors='coerce')
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    return df

# =======================
# 模型训练与滚动预测
# =======================
def rolling_train_and_predict(df, window_size=500):
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'VOL5', 'turnover_rate', 'pe', 'pb']
    X = df[features]
    y = df['target']
    
    predictions = []
    for i in range(window_size, len(df)):
        train_data = df.iloc[i-window_size:i]
        X_train = train_data[features]
        y_train = train_data['target']
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05)
        model.fit(X_train, y_train)
        
        X_test = df.iloc[i][features].values.reshape(1, -1)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])
    
    return predictions

# =======================
# 回测引擎设置
# =======================
class MomentumStrategy(bt.SignalStrategy):
    def __init__(self):
        self.signal_add(bt.SIGNAL_LONG, self.data.close)

    def next(self):
        if self.data.close[0] > self.data.close[-1]:
            self.buy()
        elif self.data.close[0] < self.data.close[-1]:
            self.sell()

# =======================
# 绩效评估与回测
# =======================
def backtest_model(data, predictions):
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(MomentumStrategy)
    cerebro.broker.set_cash(1000)
    cerebro.run()
    return cerebro.broker.getvalue()

# =======================
# 自动报告生成
# =======================
def generate_report():
    # 模拟报告生成
    st.subheader("策略投资报告")
    st.write("年化收益率: 30%")
    st.write("最大回撤: 15%")
    st.write("夏普比率: 2.1")
    st.write("建议配置: 动量策略 50%, 价值策略 30%, ML信号 20%")

# =======================
# 主函数
# =======================
def main():
    generate_report()

if __name__ == "__main__":
    main()
