import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import streamlit as st

df = pd.read_csv('./DLF.BO.csv')

df = df.drop([ 'Symbol', 'Cap', 'Adj Close'], axis=1)
df.tail()

st.title('Hello, world!')
user_input = st.text_input('Enter the ticket', 'DLF.BO')
st.write(df.describe())

st.subheader('Closing Time VS Time Chart')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel('days')
plt.ylabel('Share Price - INR')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)
