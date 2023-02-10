#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:11:32 2023

@author: Dell
"""

# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt 
import pandas as pd
#import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from streamlit_lottie import st_lottie
import requests
#from plotly import graph_objs as go
#from keras.layers import LSTM, Dense, Dropout
#from keras.models import Sequential
#from keras.preprocessing.sequence import TimeseriesGenerator
st.set_page_config(page_title="OIL FORCAST", page_icon="🖖")
import streamlit as st

def main_page():
    st.markdown("# Welcome 🎈")
    def load_lottieurl(url: str):
        r = requests.get(url)   
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://raw.githubusercontent.com/omgo101/data-science-Assignment/main/50688-oil-pump.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json,height=150,width=700)
    lottie_url ="https://raw.githubusercontent.com/omgo101/data-science-Assignment/main/66874-oil-drilling-construction.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json,height=150,width=600)
    st.markdown("""---""")
    
    st.write("Revolutionize your oil prediction capabilities with our cutting-edge machine learning model, now available on Streamlit. Our model is trained on vast amounts of oil market data, providing highly accurate predictions for future prices and trends. Streamline your analysis and make better-informed decisions with the help of our user-friendly interface. Try it now and stay ahead of the competition in the oil industry.")
    st.markdown("""----""")
    st.write("Get ahead in the oil market with our Brent oil prediction model, now live on Streamlit. Our model is trained on a large dataset of historical Brent oil prices, providing accurate predictions and insights on future trends. Stay on top of the market and make data-driven decisions with the help of our user-friendly interface. Try it now and optimize your investments in the oil industry.")
    expander = st.expander("See benefits")
    expander.write("""
- FORCAST DATA UPTO 4 YEARS
- YOU CAN FORCAST DATA AS YOU WANT IN SPECIFIC DATE RANGE
- YOU CAN DOWNLOAD CSV FILE 
- VISUALIZATION
""")

def page2():
    st.markdown("# Forcasting app")
    def load_lottieurl(url: str):
        r = requests.get(url)   
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://raw.githubusercontent.com/omgo101/data-science-Assignment/main/50688-oil-pump.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json,height=150,width=700)
    st.markdown("""---""")
    st.sidebar.markdown("# Forcasting app")
    st.markdown('**with _RSME_ :blue[11.29] and _MAE_ :blue[7.00]**')
    # Create an expander widget with a title
    expander = st.expander("Click here to see brent oil dataset")
    
    # Define the contents of the expander
    with expander:
        st.subheader('About the dataset')
        def load_data():
            return pd.read_csv('https://raw.githubusercontent.com/omgo101/dataset-/main/data11.csv')
        df = load_data()
        st.dataframe(df)
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    def load_data():
        return pd.read_csv('https://raw.githubusercontent.com/omgo101/dataset-/main/data11.csv')
    data=load_data()
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.rename(columns={'Europe Brent Spot Price FOB (Dollars per Barrel)': 'Price'}, inplace=True)
    # Predict forecast with Prophet.
    df_train = data[['Date','Price']]
    
    df_train = df_train.rename(columns={"Date": "ds", "Price": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    
    
    
    # test_data.to_csv('submission.csv', index=False)
    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.write("Forecast prices as you wnat")
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(date_range, columns=['Date'])
    def dataPreprocessing(dataFrame):
     dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])
    
     return dataFrame
    test_data = df
 
    testing_data = dataPreprocessing(test_data.copy())
  
    
    test_prediction = m.predict(pd.DataFrame({'ds':testing_data['Date']}))
    test_prediction = test_prediction['yhat']
    test_prediction = test_prediction.astype(int)
    test_data['Price'] = test_prediction
    #test_data.head()
    st.dataframe(test_data)
    
    if st.checkbox('Download CSV file'):
     test_data.to_csv('submission.csv', index=False)
     def convert_df(df):
      return df.to_csv().encode('utf-8')
     csv = convert_df(test_data)
     st.download_button(label="Download data as CSV",
     data=csv,
     file_name='forcaste_oil.csv',
     mime='text/csv')
    st.sidebar.markdown("# Main page 🎈")

def page3():
    st.markdown("# Visualization")
    def load_lottieurl(url: str):
        r = requests.get(url)   
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://raw.githubusercontent.com/omgo101/data-science-Assignment/main/50688-oil-pump.json"
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json,height=150,width=700)
    st.markdown("""---""")
    def load_data():
        return pd.read_csv('https://raw.githubusercontent.com/omgo101/dataset-/main/data11.csv')

    df = load_data()
    st.subheader("dataset")
    st.dataframe(df)
    st.subheader("Histogram")
    def load_data():
       return pd.read_csv('https://raw.githubusercontent.com/omgo101/dataset-/main/data11.csv')

    df = load_data()
# Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    filtered_data = df
    fig = plt.figure()
    plt.hist(filtered_data['Europe Brent Spot Price FOB (Dollars per Barrel)'], bins=20)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Brent crude oil price')
    st.pyplot(fig)
    st.subheader("Line Chart")
    date_min = df['Date'].min()
    date_max = df['Date'].max()
    slider = st.slider("Select a range for Date",min_value=date_min, max_value=date_max, value=(date_min.to_pydatetime(),date_max.to_pydatetime()))

   # Filter the chart data to only show data points within the selected range
    filtered_data = df[df["Date"].between(slider[0], slider[1])]

   # Create the line chart
    st.line_chart(filtered_data, x='Date', y='Europe Brent Spot Price FOB (Dollars per Barrel)')
    st.sidebar.markdown("# Visualization")

page_names_to_funcs = {
    "Main Page": main_page,
    "Forcasting app": page2,
    "Visualization": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

