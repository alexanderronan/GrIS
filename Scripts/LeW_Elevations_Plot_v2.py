# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:24:29 2023

@author: f005cb9
"""

import numpy as np
import pickle
import pandas as pd
import os
import Rbeast as rb
from datetime import datetime as dt
import time
import warnings
from matplotlib import pyplot as plt

location_coord = ["Summit", "Raven", "50km_NEEM", "NEEM", "Similar_NEEM"] #LOCATION we are cycling through
#location_coord = ["Summit"] #LOCATION we are cycling through

retracker_list = ["Elevation 1", "Elevation 2", 'Elevation 3']
#aggregation_list = ["1_12_", '1_8']
aggregation_list = ['1_8']


parameter_dict = {"Elevation 1": "Ocean","Elevation 2": "UCL-Land-Ice", "Elevation 3": "OCOG"} #Dictionary of Elevations
name_dict = {"Similar_NEEM": "Similar-NEEM","50km_NEEM": "50km-NEEM", "Similar_NEEM": "Similar-NEEM", "NEEM": "NEEM", 'Summit': 'Summit', 'Raven': 'Raven'} #Dictionary of names


#Elevation locations
beast_file_directory_non_MC_elev = #directory of level 2 elevations
output_location = #where you want to put things when you are done



metric_dict = {"LeW": "LeW (m)","TeS": "TeS (Counts/m)", "Integration": "RSI (Counts)"}
metric_title_dict = {"LeW": "LeW","TeS": "TeS", "Integration": "RSI"}
metric_error_dict =  {"LeW": "LeW","TeS": "TeS","Integration": "Int"}


#Metric locations
beast_file_directory_metric = #Beast 1b metric outputs
metric_directory = #clustered 1b metric data

pkl_ext = ".pkl"

#Function to convert datetime object to date in decimal format. Use for the "clustered" dates
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


metric = 'LeW'

    #for retracker_non_MC in retracker_list:
    
# fig = plt.figure()
# for i, label in enumerate(('A', 'B', 'C', 'D')):
#     ax = fig.add_subplot(2,2,i+1)
#     ax.text(0.05, 0.95, label, transform=ax.transAxes,
#     fontsize=16, fontweight='bold', va='top')
        
        

for site in location_coord:

    for Aggregation_interval in aggregation_list:
        
        ###########################################
        ###########################################
        
        #Plotting!
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex = True)
        fig.subplots_adjust(hspace = 0.4)
        ax1.text(0.010,0.95,'A', transform = ax1.transAxes, fontsize = 11, fontweight='bold', va='top')
        ax2.text(0.010,0.95,'B', transform = ax2.transAxes, fontsize = 11, fontweight='bold', va='top')
        ax3.text(0.010,0.95,'C', transform = ax3.transAxes, fontsize = 11, fontweight='bold', va='top')

  
        #fig.tight_layout(pad=0.5)
        
        fig.suptitle(name_dict[site])
        
        
        ###########################################
        ###########################################
        
        
        #GrIS locations
        sites_of_interest = os.listdir(beast_file_directory_non_MC_elev)
        elevation_numbers_non_MC = os.listdir('\\'.join([beast_file_directory_non_MC_elev,site]))
        
        if Aggregation_interval == "1_12_":
            non_mc_elevation_name = f'{site}_Elevation_residual_1_12th_v3.csv'
        else:
            non_mc_elevation_name = f'{site}_Elevation_residual_1_8th_v3.csv'
            
        elevation_nonmc_file = "\\".join([elevation_directory,non_mc_elevation_name])
        elevation_nonmc_df = pd.read_csv(elevation_nonmc_file)
        
        df_output_elevations_non_mc = pd.DataFrame(columns=['decimal_date_cluster', 'Elevation 1', 'Elevation 2', 'Elevation 3', 'Elevation 1 sm', 'Elevation 2 sm', 'Elevation 3 sm'])
        df_output_elevations_non_mc_beast = pd.DataFrame(columns=['beast_Elevation 1', 'beast_Elevation 2', 'beast_Elevation 3', 'decimal_date_beast_Elevation 1', 'decimal_date_beast_Elevation 2', 'decimal_date_beast_Elevation 3'])

        #########
        #Adding cluster decimal dates to output DFs 
        decimal_list = []
        for index, row in elevation_nonmc_df.iterrows():
            month,day,year = int(row['date'].split('-')[1]), int(row['date'].split('-')[2]), int(row['date'].split('-')[0])
            datetime_obj = dt(year,month,day)
            decimal = round(toYearFraction(datetime_obj), 4)
            decimal_list.append(decimal)
        
        df_output_elevations_non_mc['decimal_date_cluster'] = decimal_list
        decimal_list = []
            
        #########
        ###########################################
        ###########################################
            
        df_output_metrics = pd.DataFrame(columns=['decimal_date_cluster','LeW', 'TeS', 'Integration', 'LeW-SD', 'TeS-SD', 'Int-SD'])
        df_output_metrics_beast = pd.DataFrame(columns=['decimal_date_beast_LeW','decimal_date_beast_TeS','decimal_date_beast_Integration', 'beast_LeW', 'beast_TeS', 'beast_Integration'])
        
        metric_parameters = os.listdir('\\'.join([beast_file_directory_metric,site]))
        
        metric_file = "\\".join([metric_directory,f"cluster_{site}_LRM_without_outliers.csv"])
        metric_df = pd.read_csv(metric_file)
        
        
        #########
        #Adding cluster decimal dates to output DFs 
        decimal_list = []
        for index, row in metric_df.iterrows():
            month,day,year = int(row['date'].split('-')[0]), int(row['date'].split('-')[1]), int(row['date'].split('-')[2])
            datetime_obj = dt(year,month,day)
            decimal = round(toYearFraction(datetime_obj), 4)
            decimal_list.append(decimal)
        
        df_output_metrics['decimal_date_cluster'] = decimal_list
        decimal_list = []
        
        #########
        
        #Adding BEAST outputs to the appropriate output DF
        output_path = "\\".join([output_location,site]) #creates new folder to put the outputs
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        
        files = os.listdir("\\".join([beast_file_directory_metric,site,metric,Aggregation_interval]))
        for file in files:
            if file[-4:] == pkl_ext: 
                pkl_file = "\\".join([beast_file_directory_metric,site,metric,Aggregation_interval,file])
                with open(pkl_file, 'rb') as file:
                    x = pickle.load(file)
                df_output_metrics_beast[f'decimal_date_beast_{metric}'] = (x.time).tolist()
                df_output_metrics_beast[f"beast_{metric}"] = (x.trend.Y).tolist()
        df_output_metrics[metric] = metric_df[metric].tolist()
        df_output_metrics[f"{metric_error_dict[metric]}-SD"] = metric_df[f"Std-{metric_error_dict[metric]}"].tolist()
        
        #Set time for confidence intervals
        t2t    = np.concatenate( [x.time, np.flip(x.time)])
        
        #plotting
        
        ax1.scatter(np.array(df_output_metrics['decimal_date_cluster']),np.array(df_output_metrics[metric]),s =3, color = "k")
        upper_CI = np.add(x.trend.SD, x.trend.Y)
        lower_CI = np.subtract(x.trend.Y,x.trend.SD)
        CI = np.concatenate([upper_CI, np.flip(lower_CI)])
        ax1.fill(t2t,CI,alpha = 0.3, color = "b", linestyle = "None")
        ax1.plot(x.time,x.trend.Y, color = "b")
         
           
        ax1.errorbar(np.array(df_output_metrics['decimal_date_cluster']), np.array(df_output_metrics[metric]), yerr = np.array(df_output_metrics[f"{metric_error_dict[metric]}-SD"]), elinewidth = 0.5, capsize = 1, fmt='o', ecolor = 'k', color = 'k', markersize = '2')
     
        ax1.set_ylabel(metric_dict[metric])
        ax1.set_title("LeW (Weekly Avg. Cluster)")
        handles, labels = ax1.get_legend_handles_labels()
        order = [0,3,1,2]
        
        
        
        retracker_non_MC = "Elevation 2"
            
        files = os.listdir("\\".join([beast_file_directory_non_MC_elev,site,retracker_non_MC,Aggregation_interval]))
        for file in files:
            if file[-4:] == pkl_ext: 
                pkl_file = "\\".join([beast_file_directory_non_MC_elev,site,retracker_non_MC,Aggregation_interval,file])
                with open(pkl_file, 'rb') as file:
                    x = pickle.load(file)
                df_output_elevations_non_mc_beast[f'decimal_date_beast_{retracker_non_MC}'] = (x.time).tolist()
                df_output_elevations_non_mc_beast[f'beast_{retracker_non_MC}'] = (x.trend.Y).tolist()
        df_output_elevations_non_mc[retracker_non_MC] = elevation_nonmc_df[retracker_non_MC].tolist()
        df_output_elevations_non_mc[f"{retracker_non_MC} sm"] = elevation_nonmc_df[f"{retracker_non_MC} sm"].tolist()
        
        
        #Set time for confidence intervals
        t2t    = np.concatenate( [x.time, np.flip(x.time)])
        
        #plotting
        #axes = plt.gca()
        
        x_date  = np.array(df_output_elevations_non_mc['decimal_date_cluster'])
        y_value = np.array(df_output_elevations_non_mc[retracker_non_MC])
        
        cluster_dir = r"D:\Cryosat_2_data\Cryosat-2_Files\publication\clusters_20_km_buffer\Level-2-Elevations_v4"
        updated_elevation = "\\".join([cluster_dir,f"{site}_month_cluster_mlt_reg_NEW_ERROR_PREDICITON_v1.csv"])
        
        df_elevations_new = pd.read_csv(updated_elevation)
        if site == "Summit":
            df_elevations_new = df_elevations_new.drop([1])
        if site == "Raven":
            df_elevations_new = df_elevations_new[:-1]
        
        
        conf_interval = np.array(df_elevations_new['EV2 Conf'])
        
        ax2.scatter(x_date,y_value, s =3, color = "k", label = "Cluster")
        ax2.errorbar(x_date, y_value, yerr=conf_interval, elinewidth = 0.5, capsize = 1, fmt='o', ecolor = 'k', color = 'k', markersize = '2', label = 'CI (95%)')
        
        upper_CI = np.add(x.trend.SD, x.trend.Y)
        lower_CI = np.subtract(x.trend.Y,x.trend.SD)
        CI = np.concatenate([upper_CI, np.flip(lower_CI)])
        ax2.fill(t2t,CI,alpha = 0.3, color = "b", linestyle = "None", label = "BEAST Trend CI (95%)")
        
        if Aggregation_interval == "1_8":
            ax2.plot(x.time,x.trend.Y, color = "b", label = "BEAST Trend (1/8th Year)")
        else:
            ax2.plot(x.time,x.trend.Y, color = "b", label = "BEAST Trend (1/12th Year)")
           
        ax2.set_ylabel("Elevation (m)")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,3,2,1]
        
        ax2.set_title(f"{parameter_dict[retracker_non_MC]} Elevations (Monthly MLR Cluster)")

             
        decimal_list = []
        
        
        #####################################
        ######################
        
        retracker_non_MC = "Elevation 3"
        
        #Adding BEAST outputs to the appropriate output DF
        
        output_path = "\\".join([output_location,site]) #creates new folder to put the outputs
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        files = os.listdir("\\".join([beast_file_directory_non_MC_elev,site,retracker_non_MC,Aggregation_interval]))
        for file in files:
            if file[-4:] == pkl_ext: 
                pkl_file = "\\".join([beast_file_directory_non_MC_elev,site,retracker_non_MC,Aggregation_interval,file])
                with open(pkl_file, 'rb') as file:
                    x = pickle.load(file)
                df_output_elevations_non_mc_beast[f'decimal_date_beast_{retracker_non_MC}'] = (x.time).tolist()
                df_output_elevations_non_mc_beast[f'beast_{retracker_non_MC}'] = (x.trend.Y).tolist()
        df_output_elevations_non_mc[retracker_non_MC] = elevation_nonmc_df[retracker_non_MC].tolist()
        df_output_elevations_non_mc[f"{retracker_non_MC} sm"] = elevation_nonmc_df[f"{retracker_non_MC} sm"].tolist()
        
        
        #Set time for confidence intervals
        t2t    = np.concatenate( [x.time, np.flip(x.time)])
        
        #plotting
        #axes = plt.gca()
        
        #added
        x_date  = np.array(df_output_elevations_non_mc['decimal_date_cluster'])
        y_value = np.array(df_output_elevations_non_mc[retracker_non_MC])
        
        
       
        
        cluster_dir = #cluster directory
        updated_elevation = #elevations
        
     
        conf_interval = np.array(df_elevations_new['EV3 Conf'])
        
        
        ax3.scatter(x_date,y_value, s =3, color = "k", label = "Cluster")
        ax3.errorbar(x_date, y_value, yerr=conf_interval, elinewidth = 0.5, capsize = 1, fmt='o', ecolor = 'k', color = 'k', markersize = '2', label = 'CI (95%)')
        
        upper_CI = np.add(x.trend.SD, x.trend.Y)
        lower_CI = np.subtract(x.trend.Y,x.trend.SD)
        CI = np.concatenate([upper_CI, np.flip(lower_CI)])
        ax3.fill(t2t,CI,alpha = 0.3, color = "b", linestyle = "None", label = "BEAST Trend CI (95%)")
        
        if Aggregation_interval == "1_8":
            ax3.plot(x.time,x.trend.Y, color = "b", label = "BEAST Trend (1/8th Year)")
        else:
            ax3.plot(x.time,x.trend.Y, color = "b", label = "BEAST Trend (1/12th Year)")
           
        ax3.set_ylabel("Elevation (m)")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,3,2,1]
        
        ax3.set_title(f"{parameter_dict[retracker_non_MC]} Elevations (Monthly MLR Cluster)")
        ax3.set_xlabel("Date")

             
        decimal_list = []
            
            
        ###########################################
        ###########################################
        ###########################################
        ###########################################
        
      
            
       
        handles, labels = ax3.get_legend_handles_labels()
        order = [0,3,1,2]
        fig.subplots_adjust(bottom=0.155)
        fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = "lower center", bbox_transform=fig.transFigure, ncol=4) 
        
        #fig.tight_layout(rect=[0, 0, 0.75, 1])
        fig.savefig("\\".join([output_path,f"{site}_both_elevations_{metric}_{Aggregation_interval}_yr_v4.png"]), format = "png", dpi = 300, bbox_inches = "tight")
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
        #plt.show()
         
        decimal_list = []
        
           
                    
                    
                    
                   
