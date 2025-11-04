
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df_data = pd.read_csv('/Users/theo/Documents/CC/EV228/EV228-EDS-TD/data/CHM00054511.csv', parse_dates=['DATE'])
# df_data['PRCP'] = pd.to_numeric(df_data['PRCP'])

# df_data = df_data.dropna(subset=['DATE', 'PRCP'])
# df_data['YEAR'] = df_data['DATE'].dt.year
# df_yearly = df_data.groupby('YEAR', as_index=False)['PRCP'].mean()

# # Plot of Yearly Average

# fig, ax = plt.subplots()
# plt.plot(df_yearly['YEAR'], df_yearly['PRCP'], color='green', linewidth=1.1)
# plt.xlabel('Year')
# plt.ylabel('Precipitation (mm)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.title('Beijing Average Yearly Precipitation')
# plt.xlim(1951, 2025)
# plt.show()


# Next plot average by month
# df_data['MONTH'] = df_data['DATE'].dt.month
# df_monthly = df_data.groupby('MONTH', as_index=False)['PRCP'].mean()
# print(df_monthly)

# fig, ax = plt.subplots()
# plt.plot(df_monthly['MONTH'], df_monthly['PRCP'], color='red', label='Precipitation')
# plt.xlabel('Month')
# plt.ylabel('Precipitation, inches')
# plt.legend()
# plt.title('Beijing Average Yearly Precipitation /in')
# plt.show()

# Modularity - Defining Function
def yearly_avg(filepath, file_output, fname):
    df_data = pd.read_csv(filepath, parse_dates=['DATE'])
    df_data['PRCP'] = pd.to_numeric(df_data['PRCP'])

    df_data = df_data.dropna(subset=['DATE', 'PRCP'])
    df_data['YEAR'] = df_data['DATE'].dt.year
    df_yearly = df_data.groupby('YEAR', as_index=False)['PRCP'].mean()
    fig, ax = plt.subplots()
    plt.plot(df_yearly['YEAR'], df_yearly['PRCP'], color='green', linewidth=1.1)
    plt.xlabel('Year')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Beijing Average Yearly Precipitation')
    plt.xlim(1951, 2025)
    path_name = f"{file_output}/{fname}"
    plt.savefig(path_name, dpi=300)
    plt.show()
