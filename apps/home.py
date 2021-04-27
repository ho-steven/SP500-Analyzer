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


def app():
    st.title('ELEC7080')

    st.write("This is a sample home page.")

    # ## News Sentiment Analysis
    st.write("**Economics Events: **")
    st.table(econ_events())

    st.write('Thank you')