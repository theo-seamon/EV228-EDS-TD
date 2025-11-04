import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yearly_precip_theo as ypt
import avg_precip as ap

filepath = '/Users/theo/Documents/CC/EV228/EV228-EDS-TD/data/CHM00054511.csv'
file_output = '/Users/theo/Documents/CC/EV228/ev228_data'
fname_1 = 'beijing_yearly_avg_precip.png'
fname_2 = 'beijing_daily_precip_2025_vs_avg.png'
ypt.yearly_avg(filepath, file_output, fname_1)
ap.daily_avg(filepath, file_output, fname_2)