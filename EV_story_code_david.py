import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import xarray as xr


'''import the data'''
os.chdir(r'C:\\Users\\zheng\\Desktop\\EV228')

EVS1=pd.read_csv(r"C:\\Users\\zheng\\Desktop\\EV228\\CHM00054511.csv")

print(EVS1.head())
print(EVS1.columns)

pd.set_option('display.max_rows', 20)       
pd.set_option('display.max_columns', None)    
pd.set_option('display.width', None)          
pd.set_option('display.max_colwidth', None)   

print("\nEVS1 Preview:")
print(EVS1.head())
print("\nEVS1 Info (with NaN kept)")
print(EVS1.info())


EVS1['DATE'] = pd.to_datetime(EVS1['DATE'])

# Monthly average PRCP changing trend 
df1 = EVS1.iloc[506:].copy()
df1.set_index('DATE', inplace=True)
monthly_prcp = df1['PRCP'].resample('M').mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_prcp.index, monthly_prcp.values, color='blue', linewidth=1.3)
plt.title("Monthly Average Precipitation Trend")
plt.xlabel("Year-Month")
plt.ylabel("Precipitation (PRCP)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



#PRCP trend in 2025
df2 = EVS1.iloc[27521:].copy()
df2['RollingMean'] = df2['PRCP'].rolling(window=30, min_periods=1).mean()
df2['RollingStd'] = df2['PRCP'].rolling(window=30, min_periods=1).std()

# find anomaly threshold (mean ± 2*std)
upper_threshold = df2['RollingMean'] + 2 * df2['RollingStd']
lower_threshold = df2['RollingMean'] - 2 * df2['RollingStd']

# Identify anomalies
anomalies = df2[(df2['PRCP'] > upper_threshold) | (df2['PRCP'] < lower_threshold)]

print(f"Detected {len(anomalies)} anomalies")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df2['DATE'], df2['PRCP'], color='green', linewidth=1.1, label='Daily PRCP')
plt.plot(df2['DATE'], df2['RollingMean'], color='orange', linewidth=1.3, label='30-day Mean')
plt.fill_between(df2['DATE'], lower_threshold, upper_threshold, color='yellow', alpha=0.2, label='Normal Range')

# Highlight anomalies
plt.scatter(anomalies['DATE'], anomalies['PRCP'], color='red', s=20, label='Anomalies')

plt.title("Precipitation Anomalies 2025")
plt.xlabel("Date")
plt.ylabel("Precipitation (PRCP)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()