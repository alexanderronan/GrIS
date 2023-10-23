# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:05:20 2022

@author: Alexander Ronan
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import Rbeast as rb  #(Zhao et al. 2019
import pickle
import os



location_coord = ["50km_NEEM","NEEM","Summit","Raven","Similar_NEEM"] #SET the LOCATION we are cycling through

cluster_directory = #cluster directory
parameters = ["LeW", "Integration", "TeS"]  #Level 1B Parameters. This study focuses on LeW
#beast_dir = r"D:\Cryosat_2_data\Cryosat-2_Files\CSV_files_location\BEAST_metric_v3"
beast_dir = r"D:\Cryosat_2_data\Cryosat-2_Files\publication\BEAST_Outputs\1B_metrics"

for location in location_coord:
    
    csv = "\\".join([cluster_directory,".".join([f"cluster_{location}_without_outliers","csv"])])
    df = pd.read_csv(csv)
    
    #Define Parameters and date (series to list)
    TeS = np.array(df['TeS'])
    LeW = np.array(df['LeW'])
    test = len(LeW)
    Integration = np.array(df['Integration'])
    datestr = (df['date']).tolist()
    
    if location == "Summit":
        #Remove second value
        np.delete(TeS,1)
        np.delete(LeW,1)
        np.delete(Integration,1)
        np.delete(datestr,1)
        
    if location == "Raven":
        #Remove first and last value
        TeS = TeS[:-1]
        LeW = LeW[:-1]
        Integration = Integration[:-1]
        datestr = datestr[:-1]
    
    #set string to variable name
    myVars = vars()
    myVars["TeS"] = TeS
    myVars["LeW"] = LeW
    myVars["Integration"] = Integration
    myVars[1] = "1_"
    myVars[1/6] = "1_6"
    myVars[1/8] = "1_8"
    myVars[1/12] = "1_12_"
    myVars[1/16] = "1_16_"
    myVars[1/18] = "1_18_"
    myVars[1/21] = "1_21_"
    myVars[1/24] = "1_24_"
    myVars[1/36] = "1_36_"
    myVars[1/40] = "1_40_"
    myVars[1/48] = "1_48_"
    myVars[1/4] = "1_4"

    #Define parameter, period, and time interval for aggregation
    for parameter_name in parameters:
        period_input = 1  #in years
        aggregate_data_times = [1/4,1/6,1/8,1/12,1/16,1/18,1/21,1/24,1/36,1/40,1/48]    #in years (Aggregation Interval)

        for deltaTime_input in aggregate_data_times:   #Cycle through years
            folder_aggregate_mode = "\\".join([beast_dir,location, parameter_name,myVars[deltaTime_input]]) #creates new folder to put the outputs
            if not os.path.exists(folder_aggregate_mode):
                os.makedirs(folder_aggregate_mode)      #IF the folder doesn't exist, create it 
            #Define Metadata
            metadata = rb.args(
                isRegular = False,      #Irregularly spaced
                time = rb.args(
                    datestr = datestr,  #List of dates as strings
                    strfmt = 'mm-dd-YYYY'),     #Interprets dates
                season = 'none',     #Ignore seasonality.
                #period = period_input,
                deltaTime = deltaTime_input)
            
            #define extra parameters
            extra = rb.args(dumpInputData = True)    # make a copy of the aggregated input data in the beast ouput

            #Run the BEAST algorithm on the clustered data.
            output = rb.beast123(myVars[parameter_name], metadata, [], [], extra)
            
          
            #Decide where I want to save the BEAST output 
            output_file = "\\".join([folder_aggregate_mode, ".".join(["".join([f"output_{parameter_name}_",f"{myVars[deltaTime_input]}"]),"pkl"])])

            #Save output as pickle file, which can then be opened as a python object by another user with the correct environment.
            fileObj = open(output_file, 'wb')
            pickle.dump(output,fileObj)
            fileObj.close()
            
        
###########################################################################


