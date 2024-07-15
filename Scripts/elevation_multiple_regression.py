"""
Created on Sun Feb  5 13:01:26 2023

@author: Alexander Ronan
"""

# Importing Modules
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import netCDF4 as nt
import geopandas as geo
import datetime
import glob
import scipy.signal
import scipy
import warnings 
import math
from os.path import exists
from os import makedirs
from os import listdir
from datetime import datetime
import rasterio as rt
from datetime import timedelta
from shapely.geometry import Point
import shutil
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore')



cluster_dir = #directory of output clustered data directory
raw_dir = #directory of raw data

#Reprojection Parameters
a, b = 6378137, 6356752.3142              #a is the semimajor axis of the WGS84 ellipsoid and b is the semiminor axis of the WGS84 ellipsoid.
e = math.sqrt(1 - ((b**2)/(a**2)))       #e is the eccentricity of the WGS
lat0, lon0 = math.radians(70), math.radians(-45)       # latitude and longitude of the projected origin, and standard parallel/central meridian
mc = (math.cos(lat0) / math.sqrt(1 - ((e**2) * ((math.sin(lat0))**2))))
tc_n, tc_d = math.tan((math.pi / 4) - (lat0 / 2)), ((1 - e * math.sin(lat0)) / (1 + e * math.sin(lat0)))**(e/2)
tc = tc_n / tc_d

seconds_month = 2.628e+6


def projection(a,b,e,lat0,lon0,mc,tc,lat_inp,lon_inp):
    #Converting from WGS84 to NSIDC Stereographic North
    #https://stackoverflow.com/questions/42777086/how-to-convert-geospatial-coordinates-dataframe-to-native-x-y-projection
    lon, lat = (math.pi * lon_inp/180),  (math.pi * lat_inp/180) 
    t_n, t_d = math.tan((math.pi / 4) - (lat / 2)), ((1 - e * math.sin(lat)) / (1 + e * math.sin(lat)))**(e/2)
    t = (t_n / t_d)
    rho = (a * mc * t / tc)
    x_out, y_out = (rho * math.sin(lon - lon0)), ((-1) * rho * math.cos(lon - lon0))
        
    return x_out,y_out



loc_list = ["50km_NEEM","NEEM","Summit","Raven","Similar_NEEM"] #Locations to cycle through and their locations.

coordinate_dict = {"50km_NEEM": (-49.31994,77.20051),"NEEM": (-51.06,77.45), "Summit": (-38.4500,72.5833), "Raven":(-46.2849,66.4964), "Similar_NEEM": (-40.0,77.6)}


for location in loc_list:
    coord = coordinate_dict[location]
    lat_inp = coord[1]
    lon_inp = coord[0]
    loc = projection(a,b,e,lat0,lon0,mc,tc,lat_inp,lon_inp)
    x_test = loc[0]
    y_test = loc[1]
    
    elevation_file = "//".join([raw_dir,location,'total.csv'])  #creating directory to place elevations


    df_total = pd.read_csv(elevation_file)
    df_total.drop(['Orbit', 'Lat', 'Lon','Unnamed: 0'], axis=1, inplace=True)   #remove not-needed DF columns.
    
    
   
    #Creating dataframe to put clustered data in.
    df_cluster = pd.DataFrame(columns=['date','EV1 Pred','EV2 Pred','EV3 Pred','EV1 Conf','EV2 Conf','EV3 Conf', 'Elevation 1', 'Elevation 2', 'Elevation 3', 'EV1 c0', 'EV1 c1','EV1 c2', 'EV1 c0 SE', 'EV1 c1 SE', 'EV1 c2 SE', 'EV2 c0', 'EV2 c1','EV2 c2','EV2 c0 SE','EV2 c1 SE', 'EV2 c2 SE','EV3 c0', 'EV3 c1','EV3 c2','EV3 c0 SE','EV3 c1 SE', 'EV3 c2 SE', 'EV1 r2', 'EV2 r2', 'EV3 r2', 'EV1 pv', 'EV2 pv', 'EV3 pv', 'Proj_x', 'Proj_y', '#Samples'])

    row_list = []
    second_to_date = []
    #len(df_total)
    
    #http://inversionlabs.com/2016/03/21/best-fit-surfaces-for-3-dimensional-data.html
    
    #begin the multiple regression process + clustered data process
    for i in range(len(df_total)):
        row_list.append(i)
        first_date = row_list[0]
        try:
            if ((df_total.loc[first_date, "date"] + seconds_month) >= df_total.loc[i+1,"date"]): #The 1st through N-1 data in the weekly time frame is included
                print("normal", i)
                continue
            
            df_temp = df_total.iloc[row_list,:].mean(axis=0)
    
            elevation_1 =  df_total.loc[row_list,'Elevation 1'].tolist()
            elevation_2 = df_total.loc[row_list,'Elevation 2'].tolist()
            elevation_3 = df_total.loc[row_list,'Elevation 3'].tolist()
    
            x_pos = df_total.loc[row_list,'Proj_x'].tolist()
            y_pos = df_total.loc[row_list,'Proj_y'].tolist()
            
            retracker_list = [elevation_1, elevation_2, elevation_3]
            
            for retracker in retracker_list:
                
                #https://scipy-lectures.org/packages/statistics/auto_examples/plot_regression_3d.html
                data = pd.DataFrame({'x': x_pos, 'y': y_pos, 'z': retracker})
                model = ols("z ~ x + y", data).fit()    #Defines what linear regression model to use (OLS)
                C = model._results.params
                BSE = model._results.bse
                #evaluate elevation at specific location
                #evaluate elevation at specific location
                #https://stackoverflow.com/questions/17559408/confidence-and-prediction-intervals-with-statsmodels/47191929#47191929
                #https://github.com/statsmodels/statsmodels/issues/987
                #https://github.com/statsmodels/statsmodels/issues/4437
                #https://scipy-lectures.org/packages/statistics/auto_examples/plot_regression_3d.html
                #(ChatGPT, OpenAI, 2024) - I used this tool to investigate available OLS package tools for 3d regression prediction, and edited output code based on the other links/citations above to incorporate into my script. This was only utilized for the next four lines (and other occurences of those four lines throughout the script)
                ###################################################################################
                prediction = model.get_prediction(pd.DataFrame({'x': [x_test], 'y': [y_test]})) #same as before without z since thats the prediction!
                predictions_summary = prediction.summary_frame(alpha=0.05)
                predicted_mean = predictions_summary['mean'][0]
                predicted_mean_se = predictions_summary['mean_se'][0]
                ###################################################################################
                elevation_calc = C[1]*x_test + C[2]*y_test + C[0] #evaluate elevation 
                error_elevation = math.sqrt(((BSE[1]*x_test)**2) + ((BSE[2]*y_test)**2) + ((BSE[0])**2))  #calculated uncertainty of elevation
                if (retracker == elevation_1):
                    df_temp['Elevation 1'] = elevation_calc
                    df_temp['EV1 r2'] = model._results.rsquared_adj
                    df_temp['EV1 pv'] = model._results.f_pvalue
                    df_temp['EV1 c0'] = C[0]
                    df_temp['EV1 c1'] = C[1]
                    df_temp['EV1 c2'] = C[2]
                    df_temp['EV1 c0 SE'] = BSE[0]
                    df_temp['EV1 c1 SE'] = BSE[1]
                    df_temp['EV1 c2 SE'] = BSE[2]
                    df_temp['EV1 Pred'] = predicted_mean
                    df_temp['EV1 Conf'] = predicted_mean_se * 1.96 
                if (retracker == elevation_2):
                    df_temp['Elevation 2'] = elevation_calc
                    df_temp['EV2 r2'] = model._results.rsquared_adj
                    df_temp['EV2 pv'] = model._results.f_pvalue
                    df_temp['EV2 c0'] = C[0]
                    df_temp['EV2 c1'] = C[1]
                    df_temp['EV2 c2'] = C[2]
                    df_temp['EV2 c0 SE'] = BSE[0]
                    df_temp['EV2 c1 SE'] = BSE[1]
                    df_temp['EV2 c2 SE'] = BSE[2]
                    df_temp['EV2 Pred'] = predicted_mean
                    df_temp['EV2 Conf'] = predicted_mean_se * 1.96
                if (retracker == elevation_3):
                    df_temp['Elevation 3'] = elevation_calc
                    df_temp['EV3 r2'] = model._results.rsquared_adj
                    df_temp['EV3 pv'] = model._results.f_pvalue
                    df_temp['EV3 c0'] = C[0]
                    df_temp['EV3 c1'] = C[1]
                    df_temp['EV3 c2'] = C[2]
                    df_temp['EV3 c0 SE'] = BSE[0]
                    df_temp['EV3 c1 SE'] = BSE[1]
                    df_temp['EV3 c2 SE'] = BSE[2]
                    df_temp['EV3 Pred'] = predicted_mean
                    df_temp['EV3 Conf'] = predicted_mean_se * 1.96
                    
            number_population = len(row_list)         #Calculates number of total samples that are being averaged.
            df_temp['#Samples'] = number_population        #Adding in STD values to a temporary average df which will then be appended onto the df_average.
            date = df_total.loc[row_list,'date'].mean()
            df_temp['date'] = date
            df_cluster = df_cluster.append(df_temp, ignore_index = True)
            row_list = []
            continue
        except:
            if df_total.loc[i-1,"date"] + seconds_month >= df_total.loc[i, "date"]: #The last date in the weekly time frame is included
                print("last normal", i)
                df_temp = df_total.iloc[row_list,:].mean(axis=0)
    
                elevation_1 =  df_total.loc[row_list,'Elevation 1'].tolist()
                elevation_2 = df_total.loc[row_list,'Elevation 2'].tolist()
                elevation_3 = df_total.loc[row_list,'Elevation 3'].tolist()
                
                x_pos = df_total.loc[row_list,'Proj_x'].tolist()
                y_pos = df_total.loc[row_list,'Proj_y'].tolist()
                
                retracker_list = [elevation_1, elevation_2, elevation_3]
                
                for retracker in retracker_list:
                    data = pd.DataFrame({'x': x_pos, 'y': y_pos, 'z': retracker})
                    model = ols("z ~ x + y", data).fit()
                    C = model._results.params
                    BSE = model._results.bse
                    #evaluate elevation at specific location
                    elevation_calc = C[1]*x_test + C[2]*y_test + C[0]
                    error_elevation = math.sqrt(((BSE[1]*x_test)**2) + ((BSE[2]*y_test)**2) + ((BSE[0])**2))
                    #evaluate elevation at specific location
                    #https://stackoverflow.com/questions/17559408/confidence-and-prediction-intervals-with-statsmodels/47191929#47191929
                    #https://github.com/statsmodels/statsmodels/issues/987
                    #https://github.com/statsmodels/statsmodels/issues/4437
                    #https://scipy-lectures.org/packages/statistics/auto_examples/plot_regression_3d.html
                    #(ChatGPT, OpenAI, 2024) - I used this tool to investigate available OLS package tools for 3d regression prediction, and edited output code based on the other links/citations above to incorporate into my script. This was only utilized for the next four lines (and other occurences of those four lines throughout the script)
                
                    ###################################################################################
                    prediction = model.get_prediction(pd.DataFrame({'x': [x_test], 'y': [y_test]})) #same as before without z since thats the prediction!
                    predictions_summary = prediction.summary_frame(alpha=0.05)
                    predicted_mean = predictions_summary['mean'][0]
                    predicted_mean_se = predictions_summary['mean_se'][0]
                    ###################################################################################

                    if (retracker == elevation_1):
                    df_temp['Elevation 1'] = elevation_calc
                    df_temp['EV1 r2'] = model._results.rsquared_adj
                    df_temp['EV1 pv'] = model._results.f_pvalue
                    df_temp['EV1 c0'] = C[0]
                    df_temp['EV1 c1'] = C[1]
                    df_temp['EV1 c2'] = C[2]
                    df_temp['EV1 c0 SE'] = BSE[0]
                    df_temp['EV1 c1 SE'] = BSE[1]
                    df_temp['EV1 c2 SE'] = BSE[2]
                    df_temp['EV1 Pred'] = predicted_mean
                    df_temp['EV1 Conf'] = predicted_mean_se * 1.96 
                if (retracker == elevation_2):
                    df_temp['Elevation 2'] = elevation_calc
                    df_temp['EV2 r2'] = model._results.rsquared_adj
                    df_temp['EV2 pv'] = model._results.f_pvalue
                    df_temp['EV2 c0'] = C[0]
                    df_temp['EV2 c1'] = C[1]
                    df_temp['EV2 c2'] = C[2]
                    df_temp['EV2 c0 SE'] = BSE[0]
                    df_temp['EV2 c1 SE'] = BSE[1]
                    df_temp['EV2 c2 SE'] = BSE[2]
                    df_temp['EV2 Pred'] = predicted_mean
                    df_temp['EV2 Conf'] = predicted_mean_se * 1.96
                if (retracker == elevation_3):
                    df_temp['Elevation 3'] = elevation_calc
                    df_temp['EV3 r2'] = model._results.rsquared_adj
                    df_temp['EV3 pv'] = model._results.f_pvalue
                    df_temp['EV3 c0'] = C[0]
                    df_temp['EV3 c1'] = C[1]
                    df_temp['EV3 c2'] = C[2]
                    df_temp['EV3 c0 SE'] = BSE[0]
                    df_temp['EV3 c1 SE'] = BSE[1]
                    df_temp['EV3 c2 SE'] = BSE[2]
                    df_temp['EV3 Pred'] = predicted_mean
                    df_temp['EV3 Conf'] = predicted_mean_se * 1.96
    
                number_population = len(row_list)         #Calculates number of total samples that are being averaged.
                df_temp['#Samples'] = number_population        #Adding in STD values to a temporary average df which will then be appended onto the df_average.
                date = df_total.loc[row_list,'date'].mean()
                df_temp['date'] = date
                df_cluster = df_cluster.append(df_temp, ignore_index = True)
            else:                                   #Last one in the time series total.
                print("last not normal", i)                                      
                df_temp = df_total.iloc[row_list[:-1],:].mean(axis=0)
    
                elevation_1 =  df_total.loc[row_list[:-1],'Elevation 1'].tolist()
                elevation_2 = df_total.loc[row_list[:-1],'Elevation 2'].tolist()
                elevation_3 = df_total.loc[row_list[:-1],'Elevation 3'].tolist()
    
                x_pos = df_total.loc[row_list[:-1],'Proj_x'].tolist()
                y_pos = df_total.loc[row_list[:-1],'Proj_y'].tolist()
                
                retracker_list = [elevation_1, elevation_2, elevation_3]
    
                for retracker in retracker_list:
                    data = pd.DataFrame({'x': x_pos, 'y': y_pos, 'z': retracker})
                    model = ols("z ~ x + y", data).fit()
                    C = model._results.params
                    BSE = model._results.bse
                    #evaluate elevation at specific location
                    elevation_calc = C[1]*x_test + C[2]*y_test + C[0]
                    error_elevation = math.sqrt(((BSE[1]*x_test)**2) + ((BSE[2]*y_test)**2) + ((BSE[0])**2))
                    #evaluate elevation at specific location
                    #https://stackoverflow.com/questions/17559408/confidence-and-prediction-intervals-with-statsmodels/47191929#47191929
                    #https://github.com/statsmodels/statsmodels/issues/987
                    #https://github.com/statsmodels/statsmodels/issues/4437
                    #https://scipy-lectures.org/packages/statistics/auto_examples/plot_regression_3d.html
                    #(ChatGPT, OpenAI, 2024) - I used this tool to investigate available OLS package tools for 3d regression prediction, and edited output code based on the other links/citations above to incorporate into my script. This was only utilized for the next four lines (and other occurences of those four lines throughout the script)
                
                    ###################################################################################
                    prediction = model.get_prediction(pd.DataFrame({'x': [x_test], 'y': [y_test]})) #same as before without z since thats the prediction!
                    predictions_summary = prediction.summary_frame(alpha=0.05)
                    predicted_mean = predictions_summary['mean'][0]
                    predicted_mean_se = predictions_summary['mean_se'][0]
                    ###################################################################################
                    if (retracker == elevation_1):
                        df_temp['Elevation 1'] = elevation_calc
                        df_temp['EV1 r2'] = model._results.rsquared_adj
                        df_temp['EV1 pv'] = model._results.f_pvalue
                        df_temp['EV1 c0'] = C[0]
                        df_temp['EV1 c1'] = C[1]
                        df_temp['EV1 c2'] = C[2]
                        df_temp['EV1 c0 SE'] = BSE[0]
                        df_temp['EV1 c1 SE'] = BSE[1]
                        df_temp['EV1 c2 SE'] = BSE[2]
                        df_temp['EV1 Pred'] = predicted_mean
                        df_temp['EV1 Conf'] = predicted_mean_se * 1.96 
                    if (retracker == elevation_2):
                        df_temp['Elevation 2'] = elevation_calc
                        df_temp['EV2 r2'] = model._results.rsquared_adj
                        df_temp['EV2 pv'] = model._results.f_pvalue
                        df_temp['EV2 c0'] = C[0]
                        df_temp['EV2 c1'] = C[1]
                        df_temp['EV2 c2'] = C[2]
                        df_temp['EV2 c0 SE'] = BSE[0]
                        df_temp['EV2 c1 SE'] = BSE[1]
                        df_temp['EV2 c2 SE'] = BSE[2]
                        df_temp['EV2 Pred'] = predicted_mean
                        df_temp['EV2 Conf'] = predicted_mean_se * 1.96
                    if (retracker == elevation_3):
                        df_temp['Elevation 3'] = elevation_calc
                        df_temp['EV3 r2'] = model._results.rsquared_adj
                        df_temp['EV3 pv'] = model._results.f_pvalue
                        df_temp['EV3 c0'] = C[0]
                        df_temp['EV3 c1'] = C[1]
                        df_temp['EV3 c2'] = C[2]
                        df_temp['EV3 c0 SE'] = BSE[0]
                        df_temp['EV3 c1 SE'] = BSE[1]
                        df_temp['EV3 c2 SE'] = BSE[2]
                        df_temp['EV3 Pred'] = predicted_mean
                        df_temp['EV3 Conf'] = predicted_mean_se * 1.96
                number_population = len(row_list)         #Calculates number of total samples that are being averaged.
                df_temp['#Samples'] = number_population        #Adding in STD values to a temporary average df which will then be appended onto the df_average.
                date = df_total.loc[row_list,'date'].mean()
                df_temp['date'] = date
                df_cluster = df_cluster.append(df_temp, ignore_index = True)

    list_seconds = df_cluster['date'].tolist()
    [second_to_date.append(datetime.fromtimestamp(int(seconds)).strftime('%m-%d-%Y')) for seconds in list_seconds]
    df_cluster['date'] = second_to_date
    df_cluster.to_csv("\\".join([cluster_dir,location,"month_cluster_v13_mlt_reg.csv"]))
    second_to_date = []
