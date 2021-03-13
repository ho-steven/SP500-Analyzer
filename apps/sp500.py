import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
#import talib 
import ta
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import plotly.graph_objects as go
yf.pdr_override()


    
def user_input_features():
    today = date.today()
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker, start_date, end_date


def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


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

def get_insider(symbol):
    try:
        #symbol, start, end = user_input_features()
        

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


def get_analyst_price_targets(symbol):

    try:
        #symbol, start, end = user_input_features()

        url2 = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
        req = Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        html = soup(webpage, "html.parser")
        soup1 = soup(str(html), "html.parser")
        table = soup1.find('table', attrs={'class':'fullview-ratings-outer'})
        table_rows = table.find_all('tr')

        res = []
        for tr in table_rows:
            td = tr.find_all('td')
            row = [tr.text for tr in td] 
            res.append(row)
        new_list = [x for x in res if len(x)==5]

        analyst = pd.DataFrame(new_list, columns=["Date", "Level", "Analyst", "View", "Predition"])
        
        return analyst
        
        
    except Exception as e:
        return e




def app():
    # check if the library folder already exists, to avoid building everytime you load the pahe
    if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # install python package
    os.system(
        'pip3 install --global-option=build_ext --global-option="-L/home/appuser/lib/" --global-option="-I/home/appuser/include/" ta-lib'
    )
    # back to the cwd
    os.chdir(default_cwd)
    print(os.getcwd())
    sys.stdout.flush()

    # add the library to our current environment
    from ctypes import *

    lib = CDLL("/home/appuser/lib/libta_lib.so.0")
    # import library
    import talib
        st.write("""
    # S&P 500 Stock Analyzer
    Shown below are the **Fundamentals**, **Moving Average Crossovers**, **Bollinger Bands**, **MACD's**, **Relative Strength Indexes** of your selected stock!
    """)
    
    st.sidebar.header('User Input Parameters')
    
    # symbol, start, end = user_input_features()
    # start = pd.to_datetime(start)
    # end = pd.to_datetime(end)

    symbol, start, end = user_input_features()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    company_name = get_symbol(symbol.upper())
        

    # Read data 
    data = yf.download(symbol,start,end)
    
    # ## SMA and EMA
    #Simple Moving Average
    data['SMA'] = talib.SMA(data['Adj Close'], timeperiod = 20)

    # Exponential Moving Average
    data['EMA'] = talib.EMA(data['Adj Close'], timeperiod = 20)

    # Plot
    st.header(f"""
              Simple Moving Average vs. Exponential Moving Average\n {company_name}
              """)
    st.line_chart(data[['Adj Close','SMA','EMA']])

    # Bollinger Bands
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)

    # Plot
    st.header(f"""
              Bollinger Bands\n {company_name}
              """)
    st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

    # ## MACD (Moving Average Convergence Divergence)
    # MACD
    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Plot
    st.header(f"""
              Moving Average Convergence Divergence\n {company_name}
              """)
    st.line_chart(data[['macd','macdsignal']])

    # ## RSI (Relative Strength Index)
    # RSI
    data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

    # Plot
    st.header(f"""
              Relative Strength Index\n {company_name}
              """)
    st.line_chart(data['RSI'])


    st.write("""
    # **Fundamentals, News, Insider Trades**""")
    st.write("**Fundamental Ratios: **")
    st.dataframe(get_fundamentals(symbol))

    # ## Latest News
    st.write("**Latest News: **")
    st.table(get_news(symbol).head(5))


    # ## Recent Insider Trades
    st.write("**Recent Insider Trades: **")
    st.table(get_insider(symbol).head(5))

    # ## Recent Insider Trades
    st.write("**Analyst Ratings: **")
    st.table(get_analyst_price_targets(symbol).head(5))



    st.write("Disclaimer: The data are collected from Yahoo Finance and Finviz")


