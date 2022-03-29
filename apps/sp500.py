import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
import talib 
#import ta
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import plotly.graph_objects as go
yf.pdr_override()
from pytrends.request import TrendReq
import nltk
nltk.downloader.download('vader_lexicon')
import time
from finvizfinance.quote import finvizfinance


    
def user_input_features():
    today = date.today()
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2021-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker, start_date, end_date


def get_symbol(symbol):
    try:
        stock = finvizfinance(symbol)
        company_name = stock.ticker_fundament()
        com = list(company_name.values())[0]
        
        return com
        #url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

        #result = requests.get(url).json()
        #for x in result['ResultSet']['Result']:
            #if x['symbol'] == symbol:
                #return x['name']
    except Exception as e:
        return e
        

def get_fundamentals(symbol):
    try:
        #symbol, start, end = user_input_features()
        

        # ##Fundamentals
        url2 = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
        req = Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        
        # Find fundamentals table
        fundamentals = pd.read_html(str(html), attrs = {'class': 'snapshot-table2'})[0]
        
        # Clean up fundamentals dataframe
        fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        colOne = []
        colLength = len(fundamentals)
        for k in np.arange(0, colLength, 2):
            colOne.append(fundamentals[f'{k}'])
        attrs = pd.concat(colOne, ignore_index=True)
    
        colTwo = []
        colLength = len(fundamentals)
        for k in np.arange(1, colLength, 2):
            colTwo.append(fundamentals[f'{k}'])
        vals = pd.concat(colTwo, ignore_index=True)
        
        fundamentals = pd.DataFrame()
        fundamentals['Attributes'] = attrs
        fundamentals['Values'] = vals
        fundamentals = fundamentals.set_index('Attributes')
        return fundamentals

    except Exception as e:
        return e
    
def get_news(symbol):
    try:
        #symbol, start, end = user_input_features()
        

        url2 = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
        req = Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        # Find news table
        news = pd.read_html(str(html), attrs = {'class': 'fullview-news-outer'})[0]
        links = []
        for a in html.find_all('a', class_="tab-link-news"):
            links.append(a['href'])
        
        # Clean up news dataframe
        news.columns = ['Date', 'News Headline']
        news['Article Link'] = links
        news = news.set_index('Date')
        return news

    except Exception as e:
        return e
        
        
def news_sentiment(symbol):
    # Import libraries
    import pandas as pd
    from bs4 import BeautifulSoup
    import matplotlib.pyplot as plt
    from urllib.request import urlopen, Request
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Parameters 
    n = 5 #the # of article headlines displayed per ticker
    tickers = [symbol]

    # Get Data
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        resp = urlopen(req)    
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    try:
        for ticker in tickers:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')
        
            # print ('\n')
            # print ('Recent News Headlines for {}: '.format(ticker))
            
            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                # print(a_text,'(',td_text,')')
                if i == n-1:
                    break
    except KeyError:
        pass

    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
                
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]
            
            parsed_news.append([ticker, date, time, text])
            
    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')


    # View Data 
    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    values = []
    for ticker in tickers: 
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        # dataframe = dataframe.drop(columns = ['Headline'])
        # print ('\n')
        # print (dataframe.head())
        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)
        
    # df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment']) 
    # df = df.set_index('Ticker')
    # df = df.sort_values('Mean Sentiment', ascending=False)

    return dataframe


def get_insider(symbol):
    try:        

        url2 = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
        req = Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        # Find insider table
        insider = pd.read_html(str(html), attrs = {'class': 'body-table'})[0]
        # Clean up insider dataframe
        insider = insider.iloc[1:]
        insider.columns = ['Trader', 'Relationship', 'Date', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']
        insider = insider[['Date', 'Trader', 'Relationship', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']]
        insider = insider.set_index('Date')
        return insider

    except Exception as e:
        return e


def stock_report(symbol):
    import quantstats as qs

    # extend pandas functionality with metrics, etc.
    qs.extend_pandas()

    # fetch the daily returns for a stock
    stock = qs.utils.download_returns(symbol)
    #qs.core.plot_returns_bars(stock, "SPY")
    qs.reports.html(stock, "SPY", output="report.html")


def app():

    
    st.write("""
    # S&P 500 Stock Analyzer
    Shown below are the **Fundamentals**, **News Sentiment**, **Bollinger Bands** and **Comprehensive Report (Compared with SPY as a whole as benchmark)** of your selected stock!
       
    """)
    st.markdown("***")
    st.sidebar.header('User Input Parameters')

    symbol, start, end = user_input_features()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    #symbol1 = get_symbol(symbol.upper())
        
    #st.subheader(symbol1)

    stock = finvizfinance(symbol.lower())
    stock_chart = stock.ticker_charts()
    st.image(stock_chart)

    # Read data 
    data = yf.download(symbol,start,end,threads = False)
    
    # ## SMA and EMA
    #Simple Moving Average
    data['SMA'] = talib.SMA(data['Adj Close'], timeperiod = 20)

    # Exponential Moving Average
    data['EMA'] = talib.EMA(data['Adj Close'], timeperiod = 20)

    # Plot
    st.subheader(f"""
              Simple Moving Average vs. Exponential Moving Average\n {symbol}
              """)
    st.line_chart(data[['Adj Close','SMA','EMA']])

    # Bollinger Bands
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)

    # Plot
    st.subheader(f"""
              Bollinger Bands\n {symbol}
              """)
    st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

    # ## MACD (Moving Average Convergence Divergence)
    # MACD
    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Plot
    st.subheader(f"""
              Moving Average Convergence Divergence\n {symbol}
              """)
    st.line_chart(data[['macd','macdsignal']])

    # ## RSI (Relative Strength Index)
    # RSI
    data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

    # Plot
    st.subheader(f"""
              Relative Strength Index\n {symbol}
              """)
    st.line_chart(data['RSI'])

    st.markdown("***")

    st.subheader("Fundamentals: ")
    st.dataframe(get_fundamentals(symbol))
    st.markdown("***")   

    # ## Latest News
    st.subheader("Latest News: ")
    st.table(get_news(symbol).head(5))
    st.markdown("***") 

    # ## Recent Insider Trades
    st.subheader("Recent Insider Trades: ")
    st.table(get_insider(symbol).head(5))
    st.markdown("***") 

    st.write("Generating comprehensive stock report...")
    st.write("**please wait for some time... **")
    st.write("This section will compare the historical performance of your selected stock with SPDR S&P 500 Trust ETF (Ticker: SPY) as benchmark.")
    stock_report(symbol)
    # ## Stock report

    st.subheader(f"""**{symbol} Stock Report**""")
    
    #st.header(symbol + " Stock Report")

    HtmlFile = open("report.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    #print(source_code)
    components.html(source_code, height = 9000)

    st.write("Disclaimer: The data are collected from Google, Yahoo Finance and Finviz")


