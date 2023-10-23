"""
Created on Sat Dec 24 10:05:20 2022

@author: Alexander Ronan
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import Rbeast as rb
import pickle
import os

#LIMIT is 1/52 -> Weekly

location_coord = ["50km_NEEM","NEEM","Summit","Raven","Similar_NEEM"] #Set the location we are cycling through

#Summit remove second
#Raven remove first and last

cluster_directory = #clustered data  directory
parameters = ["Elevation 1", "Elevation 2", "Elevation 3"] #Cycle through different retrackers. 

beast_dir = #directory where the BEAST algorithm outputs are saved


for location in location_coord:
    csv = "\\".join([cluster_directory,location, 'month_cluster_v11_mlt_reg.csv'])
    df = pd.read_csv(csv)
    
    elevation_1 = (df['Elevation 1']).tolist()
    elevation_2 = (df['Elevation 2']).tolist()
    elevation_3 = (df['Elevation 3']).tolist()
    datestr = (df['date']).tolist()
    
    if location == "Summit":
        #Remove second value
        del elevation_1[1]
        del elevation_2[1]
        del elevation_3[1]
        del datestr[1]
    if location == "Raven":
        #Remove last value
        elevation_1 = elevation_1[:-1]
        elevation_2 = elevation_2[:-1]
        elevation_3 = elevation_3[:-1]
        datestr = datestr[:-1]

    #set string to variable name
    myVars = vars()
    myVars["Elevation 1"] = elevation_1
    myVars["Elevation 2"] = elevation_2
    myVars["Elevation 3"] = elevation_3
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
        #period_input = 1  #in years
        #aggregate_data_times = [1/4]    #in years
        aggregate_data_times = [1/4,1/6,1/8,1/12,1/16,1/18,1/21,1/24,1/36,1/40,1/48]    #in years

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
                season = 'none',
                #period = period_input,
                deltaTime = deltaTime_input)
            
            #define extra parameters
            extra = rb.args(dumpInputData = True)    # make a copy of the aggregated input data in the beast ouput

            #Run the BEAST
            output = rb.beast123(myVars[parameter_name], metadata, [], [], extra)
            
            #Decide where I want to save the BEAST output and plot
            output_file = "\\".join([folder_aggregate_mode, ".".join(["".join([f"output_{parameter_name}_",f"{myVars[deltaTime_input]}"]),"pkl"])])

            #Save output as pickle file!
            fileObj = open(output_file, 'wb')
            pickle.dump(output,fileObj)
            fileObj.close()
            

###########################################################################


