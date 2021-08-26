#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(['Weighted_Price'], axis=1)
df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(value=df['Close'])
df['Low'] = df['Low'].fillna(value=df['Close'])
df['Open'] = df['Open'].fillna(value=df['Close'])

print(df.head())
print(df.tail())
