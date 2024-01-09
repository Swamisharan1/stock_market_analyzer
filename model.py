import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class Stock:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.data = self.get_data()

    def get_data(self):
        data = yf.download(self.ticker_symbol, start='2020-01-01', end='2024-12-31')
        data = data.asfreq('B')
        return data

    def CurPrice(self, curDate):
        curDate = datetime.strptime(curDate, '%Y-%m-%d')
        while curDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            curDate -= timedelta(days=1)
        cur_price = self.data.loc[curDate.strftime('%Y-%m-%d'), 'Close']
        return cur_price

    def NDayRet(self, N, curDate):
        curDate = datetime.strptime(curDate, '%Y-%m-%d')
        while curDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            curDate -= timedelta(days=1)
        cur_price = self.data.loc[curDate.strftime('%Y-%m-%d'), 'Close']
        prevDate = curDate - timedelta(days=N)
        while prevDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            prevDate += timedelta(days=1)
        prev_price = self.data.loc[prevDate.strftime('%Y-%m-%d'), 'Close']
        return (cur_price - prev_price) / prev_price

    def DailyRet(self, curDate):
        curDate = datetime.strptime(curDate, '%Y-%m-%d')
        while curDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            curDate -= timedelta(days=1)
        cur_price = self.data.loc[curDate.strftime('%Y-%m-%d'), 'Close']
        prevDate = curDate - timedelta(days=1)
        while prevDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            prevDate += timedelta(days=1)
        prev_price = self.data.loc[prevDate.strftime('%Y-%m-%d'), 'Close']
        return (cur_price - prev_price) / prev_price
    def Last30daysPrice(self, curDate):
        curDate = datetime.strptime(curDate, '%Y-%m-%d')
        while curDate.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            curDate -= timedelta(days=1)
        start_date = curDate - timedelta(days=30)
        while start_date.strftime('%Y-%m-%d') not in self.data.index.strftime('%Y-%m-%d'):
            start_date += timedelta(days=1)
        prices_30d = self.data.loc[start_date.strftime('%Y-%m-%d'):curDate.strftime('%Y-%m-%d'), 'Close']
        return prices_30d

    def forecast(self, steps):
        model = ARIMA(self.data['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast= model_fit.forecast(steps=steps)
        return forecast

class Portfolio:
    def __init__(self, stocks):
        self.stocks = stocks

    def CAGR(self, V_begin, V_final, t):
        return ((V_final / V_begin) ** (1 / t)) - 1

    def volatility(self, daily_returns):
        return np.sqrt(252) * np.std(daily_returns)

    def sharpe_ratio(self, daily_returns):
        return np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)

    def select_stocks(self, N, curDate, initial_equity):
        selected_stocks = []
        for ticker, stock in self.stocks.items():
            return_N_days = stock.NDayRet(N, curDate)
            if return_N_days > 0:
                selected_stocks.append(ticker)
        
        # Calculate the number of shares to buy for each selected stock
        num_stocks = len(selected_stocks)
        if num_stocks > 0:
            equity_per_stock = initial_equity / num_stocks
            shares_to_buy = {ticker: equity_per_stock / self.stocks[ticker].CurPrice(curDate) for ticker in selected_stocks}
        else:
            shares_to_buy = {}

        return shares_to_buy

    def rebalance_portfolio(self, start_date, end_date):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        portfolio = {}
        curDate = start_date
        while curDate <= end_date:
            selected_stocks = self.select_stocks(curDate.strftime('%Y-%m-%d'))
            portfolio[curDate.strftime('%Y-%m-%d')] = selected_stocks
            if curDate.month == 12:
                curDate = curDate.replace(year=curDate.year + 1, month=1)
            else:
                curDate = curDate.replace(month=curDate.month + 1)
        return portfolio

def main():
    st.title("Stock Portfolio Performance")
    start_date = st.date_input('Simulation Start Date')
    end_date = st.date_input('Simulation End Date')
    num_days = st.number_input('Number of Days to Measure Performance', min_value=1, value=30)
    initial_equity = st.number_input('Initial Equity', min_value=0.0, value=1000000.0)

    if st.button('Run Simulation'):
        nifty_50_tickers = ['ACC.NS', 'ADANIPORTS.NS', 'AMBUJACEM.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS']
        stocks = {}
        for ticker in nifty_50_tickers:
            stocks[ticker] = Stock(ticker)
        portfolio = Portfolio(stocks)
        N = num_days 
        selected_stocks = portfolio.select_stocks(N, str(start_date), initial_equity)
        st.subheader('Selected Stocks & Quantity')
        st.write(selected_stocks)

        nifty50 = yf.download('^NSEI', start=str(start_date), end=str(end_date))
        nifty_cagr = portfolio.CAGR(nifty50.iloc[0]['Close'], nifty50.iloc[-1]['Close'], 3)
        nifty_volatility = portfolio.volatility(nifty50['Close'].pct_change().dropna())
        nifty_sharpe_ratio = portfolio.sharpe_ratio(nifty50['Close'].pct_change().dropna())
        benchmark = pd.concat([stock.data['Close'] for stock in stocks.values()], axis=1).mean(axis=1)
        benchmark_cagr = portfolio.CAGR(benchmark.iloc[0], benchmark.iloc[-1], 3)
        benchmark_volatility = portfolio.volatility(benchmark.pct_change().dropna())
        benchmark_sharpe_ratio = portfolio.sharpe_ratio(benchmark.pct_change().dropna())
        sample_strategy = pd.concat([stock.data['Close'] for ticker, stock in stocks.items() if ticker in portfolio.select_stocks(N, str(start_date), initial_equity)], axis=1).mean(axis=1)
        sample_strategy_cagr = portfolio.CAGR(sample_strategy.iloc[0], sample_strategy.iloc[-1], 3)
        sample_strategy_volatility = portfolio.volatility(sample_strategy.pct_change().dropna())
        sample_strategy_sharpe_ratio = portfolio.sharpe_ratio(sample_strategy.pct_change().dropna())
        performance_summary = pd.DataFrame({
            'CAGR': [nifty_cagr, benchmark_cagr, sample_strategy_cagr],
            'Volatility': [nifty_volatility, benchmark_volatility, sample_strategy_volatility],
            'Sharpe Ratio': [nifty_sharpe_ratio, benchmark_sharpe_ratio, sample_strategy_sharpe_ratio]
        }, index=['Nifty Index', 'Benchmark Allocation', 'Sample Strategy'])
        st.subheader('Performance Metrics')
        st.write(performance_summary)

        nifty50_normalized = nifty50['Close'] / nifty50.iloc[0]['Close']
        benchmark_normalized = benchmark / benchmark.iloc[0]
        sample_strategy_normalized = sample_strategy / sample_strategy.iloc[0]

        st.subheader('Equity Curves')
        st.line_chart({
            'Nifty Index': nifty50_normalized,
            'Benchmark Allocation': benchmark_normalized,
            'Sample Strategy': sample_strategy_normalized
        })


        performance_summary = pd.DataFrame({
            'CAGR': [nifty_cagr, benchmark_cagr, sample_strategy_cagr],
            'Volatility': [nifty_volatility, benchmark_volatility, sample_strategy_volatility],
            'Sharpe Ratio': [nifty_sharpe_ratio, benchmark_sharpe_ratio, sample_strategy_sharpe_ratio]
        }, index=['Nifty Index', 'Benchmark Allocation', 'Sample Strategy'])


if __name__ == "__main__":
    main()
