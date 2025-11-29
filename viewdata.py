import pandas as pd
df = pd.read_csv('weather_data.csv')
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)