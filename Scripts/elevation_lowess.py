"""
Created on Thu Feb 16 15:45:12 2023

@author: Alexander Ronan
"""
import pandas as pd
import numpy as np
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


location_coord = ["NEEM","Summit","Raven","Similar_NEEM","50km_NEEM"] #SET the LOCATION
#location_coord = ['Summit']
cluster_dir = #clustered data directory

retrackers = ['Elevation 1', 'Elevation 2', 'Elevation 3']  #List of retrackers to cycle through

elevation_residuals_dir = #Directory to put the results

epoch_time = datetime(1970, 1, 1) #Setting date to set seconds to zero.

#https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
lowess = sm.nonparametric.lowess    #Define the LOWESS and Use it!
for location in location_coord:
    output = pd.DataFrame(columns = ['date', 'Elevation 1', 'Elevation 1 sm', 'Elevation 2', 'Elevation 2 sm', 'Elevation 3', 'Elevation 3 sm'])
    df = pd.read_csv("\\".join([cluster_dir,location,"month_cluster_v11_mlt_reg.csv"]))
    if location == "Summit":
        df = df.drop([1])
    if location == "Raven":
        df = df[:-1]
    dates = df['date'].tolist()
    seconds_list = []
    dates_list = []
    for date in dates:  #Convert date to seconds.
        date_obj = datetime(int(date.split('-')[2]), int(date.split('-')[0]), int(date.split('-')[1]), 0, 0)
        delta = (date_obj - epoch_time)
        delta = delta.total_seconds()
        seconds_list.append(delta)
        dates_list.append(date_obj)
    seconds_array = np.array(seconds_list)
    output['date'] = dates_list
    date_array = np.array(dates_list)
    for retracker in retrackers:
        elevation= np.array(df[retracker].tolist())
        output[retracker] = elevation
        z = lowess(elevation, seconds_array, frac=(1/12))   #Set the fraction (1/12th year or 1/8th year for this study)
        output[f'{retracker} sm'] = z[:,1]   #output smoothed elevations from lowess to DF
        #plt.plot(date_array, elevation, label = f'{retracker}')
        #plt.plot(date_array, z[:,1], label = f'{retracker}-lowess')
        
    # plt.xlabel('Date (s since 1/1/1970)')
    # plt.ylabel('Elevation (m)')
    # plt.legend()
    # plt.title(f"Elevation at {location}")
    # plt.savefig("\\".join([elevation_residuals_dir,f"{location}_Elevation.png"]))
    # plt.clf()
    
    output.to_csv("\\".join([elevation_residuals_dir,f"{location}_Elevation_residual_1_12th_v1.csv"])) #save as CSV file. 
    #output.to_csv("\\".join([elevation_residuals_dir,f"{location}_Elevation_residual_1_8th_v1.csv"]))

    
    
    
