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
