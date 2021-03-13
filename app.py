import streamlit as st
from multiapp import MultiApp
from apps import home, sp500 # import your app modules here


app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("S&P 500 Stock Analyzer", sp500.app)


# The main app
app.run()
