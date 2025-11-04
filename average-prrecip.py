import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import xarray as xr

# ds = pd.read_csv(r"C:\\Users\\zheng\\Desktop\\EV228\\CHM00054511.csv")
ds = pd.read_csv('/Users/theo/Documents/CC/EV228/EV228-EDS-TD/data/CHM00054511.csv', parse_dates=['DATE'])
ds['PRCP'] = pd.to_numeric(ds['PRCP'])
ds['DATE'] = pd.to_datetime(ds['DATE'])
ds_sort = ds
print(ds)
ds_sort['DATE'] = ds['DATE'].dt.day_of_year
df_daily = ds_sort.groupby('DATE', as_index= False)['PRCP'].mean()


ds1 = ds.iloc[27520:].copy()  # 2025 Data
plt.figure(figsize=(12, 6))
plt.plot(ds1['DATE'], ds1['PRCP'], color='green', linewidth=1.3, label ='2025 Precipitation')
plt.plot(df_daily['DATE'], df_daily['PRCP'], color='black', linewidth=0.5, label = 'Average Precipitation (1951-2025)')
plt.title("Beijing: 2025 Precpitation vs Average Precipitation (1951-2025)")
plt.xlabel("Days in 2025")
plt.ylabel("Precipitation (mm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
print(ds1)