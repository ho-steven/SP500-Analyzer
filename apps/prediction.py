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
from pandas_datareader import data
import investpy


    
def user_input_features():
    today = date.today()
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2020-01-01')
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
        
def technical(symbol):
    ta = investpy.search_quotes(text=symbol, products=['stocks'], countries=['united states'],n_results=1)
    today = datetime.today().strftime('%d/%m/%Y')
    t_indicator_1d = ta.retrieve_technical_indicators(interval="daily")
    t_indicator_1d.to_csv("ta_1d.csv",index=False)    


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


def lstm(symbol, start_date, end_date):
    yf.pdr_override()

    df_X = data.DataReader(symbol,start=start_date, end=end_date, data_source='yahoo')
    df_y = df_X[:1500]['Close']
    df_X.drop(df_X.columns[len(df_X.columns)-1], axis=1, inplace=True)    
    df_X['TradeDate']=df_X.index
    #fig = df_X.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)
    #st.pyplot(fig)

    # Extracting the closing prices of each day
    FullData=df_X[['Close']].values
    #print(FullData[0:5])
    
    # Feature Scaling for fast training of neural networks
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Choosing between Standardization or normalization
    #sc = StandardScaler()
    sc=MinMaxScaler()
    
    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)

    # split into samples
    X_samples = list()
    y_samples = list()

    NumerOfRows = len(X)
    TimeSteps=10  # next day's Price Prediction is based on last how many past day's prices

    # Iterate thru the values to create combinations
    for i in range(TimeSteps , NumerOfRows , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i]
        X_samples.append(x_sample)
        y_samples.append(y_sample)

    ################################################
    # Reshape the Input as a 3D (number of samples, Time Steps, Features)
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    #print('\n#### Input Data shape ####')
    #print(X_data.shape)

    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data=np.array(y_samples)
    y_data=y_data.reshape(y_data.shape[0], 1)

    # Choosing the number of testing data records
    TestingRecords=5

    # Splitting the data into train and test
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    y_train=y_data[:-TestingRecords]
    y_test=y_data[-TestingRecords:]

    # Defining Input shapes for LSTM
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

    # Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

    # Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))


    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################

    import time
    # Measuring the time taken by the model to train
    StartTime=time.time()

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

    EndTime=time.time()
    st.write("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')

    # Making predictions on test data
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)

    # Getting the original price values for testing data
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)

    # Accuracy of the predictions
    st.write('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())

    # Visualising the results
    import matplotlib.pyplot as plt

    plt.plot(predicted_Price, color = 'blue', label = 'Predicted Volume')
    plt.plot(orig, color = 'lightblue', label = 'Original Volume')

    plt.title('Stock Price Predictions')
    plt.xlabel('Trading Date')
    plt.xticks(range(TestingRecords), df_X.tail(TestingRecords)['TradeDate'])
    plt.ylabel('Stock Price')

    plt.legend()
    fig1=plt.gcf()
    fig1.set_figwidth(20)
    fig1.set_figheight(6)
    st.pyplot(fig1)
    #fig1.savefig("predit5.png")
    #from PIL import Image
    #image = Image.open('predit5.png')

    #st.image(image, caption='Predition on Testing Set')

    # Generating predictions on full data
    TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
    TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))

    FullDataPredictions=np.append(TrainPredictions, TestPredictions)
    FullDataOrig=FullData[TimeSteps:]

    # plotting the full data
    plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
    plt.plot(FullDataOrig , color = 'lightblue', label = 'Original Price')
    plt.title('Stock Price Predictions')
    plt.xlabel('Trading Date')
    plt.ylabel('Stock Price')
    plt.legend()
    fig=plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(8)
    st.pyplot(fig)
    #plt.show()
    #fig.savefig("preditfull.png")
    #from PIL import Image
    #image = Image.open('preditfull.png')
    #st.image(image, caption='Predition for the whole period')

    # Last 10 days prices
    Last10Days = df_X['Close'].tail(10).to_numpy()

    # Normalizing the data just like we did for training the model
    Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))

    # Changing the shape of the data to 3D
    # Choosing TimeSteps as 10 because we have used the same for training
    NumSamples=1
    TimeSteps=10
    NumFeatures=1
    Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

    #############################

    # Making predictions on data
    import sys
    predicted_Price = regressor.predict(Last10Days)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    #final_predit = np.savetxt(sys.stdout, predicted_Price, fmt="%.3f")
    final_predit = "".join(str(x) for x in predicted_Price)
    st.write("The next day price prediction for our LSTM Model is: ", final_predit)


def app():

    
    st.header("Financial Forecasting for S&P 500 Stocks")
    st.write("This section will showcase our LSTM model price prediction, Technical Analysis, News Sentiment Analysis, Analyst's Ratings of your chosen stock.")
    st.write("**Disclaimer: Historical data instead of real-time data are used in this section. We do not ensure the accuracy of our prediction model and shall not be liable for any financial loss for your investment.")
    st.markdown("***")

    
    st.sidebar.header('User Input Parameters')
    symbol, start, end = user_input_features()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    #symbol1 = get_symbol(symbol.upper())
        
    #st.header(f""" {symbol1} """)

    stock = finvizfinance(symbol.lower())
    stock_chart = stock.ticker_charts()
    st.image(stock_chart)

    # ## Technical Analysis
    st.subheader("Technical Forecast: ")    
    technical(symbol)
    ta_1d = pd.read_csv("ta_1d.csv")
    st.dataframe(ta_1d)
    st.markdown("***")

    # ## News Sentiment Analysis
    st.subheader("News Sentiment Analysis: ")
    st.table(news_sentiment(symbol).head(5))
    st.markdown("***")

    # ## Analyst Ratings
    st.subheader("Analyst Ratings: ")
    st.table(get_analyst_price_targets(symbol))
    st.markdown("***")
    
    # ##LSTM
    st.subheader("LSTM Model Prediction: ")
    st.write("This section will show how LSTMs can be used to learn the patterns in the stock prices. Using this model you will be able to predict tomorrow’s price of your stock based on the last 10 days prices.")
    st.write("It will take around 5 minutes to generate the result. Please be patient.")
    lstm(symbol,start,end)
    st.markdown("***")    

    st.write("Disclaimer: The data are collected from Google, Yahoo Finance and Finviz")


