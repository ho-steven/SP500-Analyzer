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
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker, start_date, end_date


def get_symbol(symbol):
    stock = finvizfinance(symbol)
    company_name = stock.ticker_fundament()
    com = list(company_name.values())[0]
    
    return com
    #url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    #result = requests.get(url).json()
    #for x in result['ResultSet']['Result']:
        #if x['symbol'] == symbol:
            #return x['name']


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


  
    
    # print ('\n')
    # print (df)



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


def get_google_trends(symbol):
    # set US timezone
    pytrends = TrendReq(hl = 'en-US', tz = 360)

    pytrends.trending_searches(pn = 'united_states')

    # kw_list  keyworld list 
    kw_list = [symbol + ' stock']

    # 然後就可以透過 build_payload 建立查詢作業，再用 interest_over_time() 呈現數據
    # 其中的　timeframe 參數很重要！它會改變你的數據格式
    #  填入 'all' 就是全期資料，資料會以月頻率更新；
    #  填入 'today 5-y' 就是至今的五年，只能設定 5 年，資料會以週頻率更新；
    #  填入 'today 3-m' 就是至今的三個月，只能設定 1,2,3 月，資料會以日頻率更新；
    #  填入 'now 7-d' 就是至今的七天，只能設定 1,7 天，資料會以小時頻率更新；
    #  填入 'now 1-H' 就是至今的一小時，只能設定 1,4 小時，資料會以每分鐘頻率更新；
    #  填入 '2015-01-01 2019-01-01' 就是 2015 年初至 2019 年初
    pytrends.build_payload(kw_list, timeframe = 'today 3-m')
    pytrends.interest_over_time()

    trend_data=pytrends.interest_over_time()
    return trend_data


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
    Shown below are the **Fundamentals**, **News Sentiment**, **Bollinger Bands**, **Analyst Ratings**, **Google Search Trends** and **Comprehensive Report (Compared with SPY as a whole as benchmark)** of your selected stock!
       
    """)


    
    st.sidebar.header('User Input Parameters')
    
    # symbol, start, end = user_input_features()
    # start = pd.to_datetime(start)
    # end = pd.to_datetime(end)

    symbol, start, end = user_input_features()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    symbol1 = get_symbol(symbol.upper())
        
    st.header(f"""** {symbol1} **""")

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
    st.header(f"""
              Simple Moving Average vs. Exponential Moving Average\n {symbol}
              """)
    st.line_chart(data[['Adj Close','SMA','EMA']])

    # Bollinger Bands
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)

    # Plot
    st.header(f"""
              Bollinger Bands\n {symbol}
              """)
    st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

    # ## MACD (Moving Average Convergence Divergence)
    # MACD
    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Plot
    st.header(f"""
              Moving Average Convergence Divergence\n {symbol}
              """)
    st.line_chart(data[['macd','macdsignal']])

    # ## RSI (Relative Strength Index)
    # RSI
    data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

    # Plot
    st.header(f"""
              Relative Strength Index\n {symbol}
              """)
    st.line_chart(data['RSI'])


    st.write("""
    # **Fundamentals, News, Insider Trades**""")
    st.write("**Fundamental Ratios: **")
    st.dataframe(get_fundamentals(symbol))

    # ## Latest News
    st.write("**Latest News: **")
    st.table(get_news(symbol).head(5))

    # ## News Sentiment Analysis
    st.write("**News Sentiment Analysis: **")
    st.table(news_sentiment(symbol).head(5))

    # ## Recent Insider Trades
    st.write("**Recent Insider Trades: **")
    st.table(get_insider(symbol).head(5))

    # ## Analyst Ratings
    st.write("**Analyst Ratings: **")
    st.table(get_analyst_price_targets(symbol))
    
    # # ## Past 3 months Google Search Trends
    # st.write("**Google Search Trends: **")
    # st.line_chart(get_google_trends(symbol))    

    st.write("Generating comprehensive stock report...")
    st.write("**please wait for some time... **")
    stock_report(symbol)
    # ## Stock report

    st.write(f"""
    # **{symbol} Stock Report**""")
    
    #st.header(symbol + " Stock Report")

    HtmlFile = open("report.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    #print(source_code)
    components.html(source_code, height = 9000)

    st.write("Disclaimer: The data are collected from Google, Yahoo Finance and Finviz")


