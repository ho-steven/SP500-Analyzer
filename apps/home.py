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
    st.title('ELEC7080')

    st.write("The Robo-advisor and Smart Investment Fund is a system that focuses on the analysis of the Standard & Poor's 500 (S&P 500) index. The system is composed of three essential parts, namely:")

    st.write("**1.	Pre-trade Analysis and Data Acquisition**")
    st.write("**2.	Trade Execution with Trading strategies, Artificial Intelligence and Backtesting models**")
    st.write("**3.	Post-trade Risk Management and Control**")
    st.markdown("***")
    st.write("**Our aim:**")
    st.write("Using Machine Learning, Artificial Intelligence combine with statistic knowledge to generate the insight towards the market. We analyse our data, use AI techniques to construct trading strategies and perform backtesting to balance the risks and returns for investments.")

    st.markdown("***")
    st.write("**System Flow Diagram:**")
    from PIL import Image
    image = Image.open('intro1.jpg')
    st.image(image)
    st.markdown("***")
    st.write("**Development timeline:**")
    image1 = Image.open('time.jpg')
    st.image(image1)
    st.markdown("***")
    st.write("""
    # **Pre-trade Analysis**""")
    st.write("Our Pre-trade analysis covers six important aspects:")    
    st.write("1.**(Macroeconomics)** Major economic events/ development")
    st.write("2.**(Market)** S&P 500 market analysis")
    st.write("3.**(Technical)** Technical Indicators")
    st.write("4.**(Fundamentals)** Balance sheets analysis")    
    st.write("5.**(News)** News analysis and sentiment analysis")
    st.write("6.**(Analysts/KOLs)** Analyst ratings/ Insiders trading")
    st.markdown("***")
    # ## Econ calender
    st.write("**Economic Calender: **")
    st.table(econ_events())
    st.markdown("***")
    st.write("**World Indexes Correlation Matrix: **")
    image2 = Image.open('corr.png')
    st.image(image2)

    st.markdown("***")
    st.write("**S&P 500 K-means clustering:**")
    image3 = Image.open('cluster.png')
    st.image(image3)    
    st.dataframe(cluster())




