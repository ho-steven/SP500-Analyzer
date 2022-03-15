import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
import talib 
import matplotlib.ticker as mticker
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as soup
import urllib.request
from urllib.request import Request, urlopen
import plotly.graph_objects as go
import pytrends


def econ_events():

    url = 'https://investing.com/economic-calendar/'
    r = urllib.request.Request(url)
    r.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36')
    response = urllib.request.urlopen(r)
    soup = BeautifulSoup(response.read(), 'html.parser')

    # find the target table for the data
    table = soup.find('table', {'id': 'economicCalendarData'})
    content = table.find('tbody').findAll('tr', {'class': 'js-event-item'})

    # get things in dictionary, append to result
    result = []
    for i in content:
        news = {'time': None,
                'country': None,
                'impact': None,
                'event': None,
                'actual': None,
                'forecast': None,
                'previous': None}
        
        news['time'] = i.attrs['data-event-datetime']
        news['country'] = i.find('td', {'class': 'flagCur'}).find('span').get('title')
        news['impact'] = i.find('td', {'class': 'sentiment'}).get('title')
        news['event'] = i.find('td', {'class': 'event'}).find('a').text.strip()
        news['actual'] = i.find('td', {'class': 'bold'}).text.strip()
        news['forecast'] = i.find('td', {'class': 'fore'}).text.strip()
        news['previous'] = i.find('td', {'class': 'prev'}).text.strip()
        result.append(news)
            
    event_df = pd.DataFrame.from_dict(result)
    event_df = event_df[list(event_df.columns)[-1:] + list(event_df.columns)[:-1]]
    
    return event_df


def cluster():
    from pylab import plot,show
    from numpy import vstack,array
    from numpy.random import rand
    import numpy as np
    from scipy.cluster.vq import kmeans,vq
    import pandas as pd
    import pandas_datareader as dr
    from math import sqrt
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt

    read_df = pd.read_csv('avg_return_volatilities.csv', index_col=[0])
    data_df = pd.DataFrame(read_df)

    data_df.drop('ENPH',inplace=True)
    data_df.drop('TSLA',inplace=True)
    data_df.drop('VIAC',inplace=True)
    data_df.drop('AMD',inplace=True)
    data_df.drop('CARR',inplace=True)
    data_df.drop('ETSY',inplace=True)

    #recreate data to feed into the algorithm
    data = np.asarray([np.asarray(data_df['Returns']),np.asarray(data_df['Volatility'])]).T

    # computing K-Means with K = 8 (8 clusters)
    centroids,_ = kmeans(data,8)
    # assign each sample to a cluster
    idx,_ = vq(data,centroids)
    # some plotting using numpy's logical indexing
    plot(data[idx==0,0],data[idx==0,1],'ob',
        data[idx==1,0],data[idx==1,1],'oy',
        data[idx==2,0],data[idx==2,1],'or',
        data[idx==3,0],data[idx==3,1],'og',
        data[idx==4,0],data[idx==4,1],'om',
        data[idx==5,0],data[idx==5,1],'oc',
        data[idx==6,0],data[idx==6,1],'xg',
        data[idx==7,0],data[idx==7,1],'xr'    
        
        )
    plot(centroids[:,0],centroids[:,1],'sk',markersize=8)

    details = [(name,cluster) for name, cluster in zip(data_df.index,idx)]
    #for detail in details:
    #    print(detail)

    df = pd.DataFrame(details, columns = ['Stock', 'Cluster'])

    return df





def app():
    st.title('COMP7409 Machine Learning in Trading and Finance - Group K')
    st.subheader("Project topic: Financial forecasting by Machine Learning")
    st.markdown("***")

    st.subheader("Project Objectives")
    st.write("Stock price prediction and financial forecasting have been attractive research topics for both researchers and investors. Modern deep learning techniques such as long short-term memory (LSTM) using historical stock price data are popular in financial forecasting.")
    st.write("This project has two main parts: First, to conduct literature review from previous research papers regarding the stock prediction model using LSTM neural network. For example, to summarize the research findings in terms of methodology, use of data sources and performance accuracy, additionally to identify macro-/micro-factors as underlying approach and limitations from existing models. Second part is to build a web dashboard stock screening platform using the web scraping techniques introduced in the lecture. This project has a significant contribution to literature as there was no research extensively specialized on building a hybrid, comprehensive and customized stock analyser with price prediction.")
    st.markdown("***")

    st.subheader("System Flow Diagram:")
    from PIL import Image
    image = Image.open('intro1.jpg')
    st.image(image)
    st.markdown("***")

    st.subheader("Literature review")
    st.write("LSTMs are widely used for sequence prediction problems and have proven to be effective as it can store important past information and forget those not required. We will summarize the findings from the selected papers which used LSTM neural network for financial forecasting, thus, to make assumptions and explore a methodology to examine the results and performance accuracy in terms of stock price movements supported by predefined macro- and micro-economic factors.")
    st.markdown("**_D. Nelson, A. Pereira, R. Oliveira. (2017) Stock market’s price movement prediction with LSTM neural networks, International Joint Conference on Neutral Works (IJCNN), pp.1419-1426_**")
    st.markdown("**_M. Adil, H. Mhamed. (2020) Stock Market Prediction Using LSTM Recurrent Neural Network, Procedia Computer Science, Vol.170, p.1168-1173_**")
    st.markdown("**_D. Duemig. (2019) Predicting stock prices with LSTM Networks, CS230, Stanford University, California_**")
    st.markdown("***")

    st.subheader("Deliverables")
    st.write("Inspired by the above papers, this project aims to strengthen techniques from the LSTM machine learning model using Standard & Poor's 500 (S&P 500) stocks as reference.  In addition to predicting S&P 500 stock price movements based on historical stock price data, the result would be visualized on a web dashboard stock screening platform.")
    st.markdown("**1. Enhanced experiment design**")
    st.write("Explore a method to develop an enhanced experiment design by improving the model with pre-processing data which would obtain a more accurate trained algorithm and maximize the prediction accuracy.")
    st.markdown("**2. Multiple factors and non-technical feature analysis**")
    st.write("Build a hybrid analyser by inputting non-technical features and multiple factors that impact stock performance, to investigate and guide future work in determining model architectures and features sets.")
    st.markdown("***")    
    
    st.subheader("Pre-trade Analysis")
    st.write("Our Pre-trade analysis covers six important aspects:")    
    st.write("1.**(Macroeconomics)** Major economic events/ development")
    st.write("2.**(Market)** S&P 500 market analysis")
    st.write("3.**(Technical)** Technical Indicators")
    st.write("4.**(Fundamentals)** Balance sheets analysis")    
    st.write("5.**(News)** News analysis and sentiment analysis")
    st.write("6.**(Analysts/KOLs)** Analyst ratings/ Insiders trading")
    st.markdown("***")
    # ## Econ calender
    st.subheader("Economic Calender:")
    st.table(econ_events())
    st.markdown("***")
    st.subheader("World Indexes Correlation Matrix:")
    image2 = Image.open('corr.png')
    st.image(image2)

    st.markdown("***")
    st.subheader("S&P 500 K-means clustering:")
    image3 = Image.open('cluster.png')
    st.image(image3)    
    st.dataframe(cluster())




