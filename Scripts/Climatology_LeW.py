# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:14:08 2024

@author: f005cb9
"""

import pandas as pd
import scipy as sc
import numpy as np
import datetime

file  = #File with Dates converted from seconds since 1-1-1970 to Present.

months = list(range(1,13))


df = pd.read_csv(file)
date_array = np.array(df['date'])
lew_array = np.array(df['LeW'])
month_array = np.empty(12)
final_average_per_month_leW = np.empty(12)
df_averages = pd.DataFrame(columns=['Month', 'Mean_LeW', 'STD'])

for month in months:
    month_counter = 0
    count = 0
    for i in range(len(date_array)):
        month_in_array = int(datetime.datetime.strptime(date_array[i],'%Y-%m-%d').month) #for RAW DATA
        #month_in_array = int(date_array[i].split("-")[0]) #For Clustered Data
        if  month_in_array == month:
            count = count + 1
        else:
            continue
    
    LeW_month_array = np.empty(count)
    
    count = 0 
    for i in range(len(date_array)):
        month_in_array = int(datetime.datetime.strptime(date_array[i],'%Y-%m-%d').month) # for RAW DATA
        #month_in_array = int(date_array[i].split("-")[0]) #For Clustered Data
        if month_in_array == month:
            LeW_month_array[count] = lew_array[i]
            count = count + 1
        else:
            continue
        
    count = 0
    print(f'The average LeW for {month} is {LeW_month_array.mean()} and STD of {LeW_month_array.std()}')
    df_averages.loc[f'{month}'] = pd.Series({'Month': month, 'Mean_LeW': LeW_month_array.mean(),'STD':LeW_month_array.std() })
    month_counter = month_counter + 1
    
    df_averages.to_csv()

            
