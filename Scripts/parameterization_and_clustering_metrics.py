Created on Sat Jan  7 13:00:58 2023

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
warnings.filterwarnings('ignore')

#Create variables year_range and month_range to circle through everything, with month/day criteria
days_thirty, days_thirty_one, leap_year  = ['4','6','9','11'], ['1','3','5','7','8','10','12'], ['2016', '2020']

#File Directories
lrm_1b_directory =  #Location of LRM 1b hdr and nc files
sarin_1b_directory = #Location of SARin 1b hdr and nc files

#Raw file directories
raw_dir = #location of calculated metrics per location.


#Cluster directories
cluster_dir = #Location of where to put output csv files containing the clustered metric data.


#Total Directories
totals_dir = #Location of of where to put a "total" directory.


file_structure = #File Structure that will be inserted at each location prior to CSV writing.
cryosat_dir = #Directory with everything in it.
ext = '.csv' #CSV extension
gimp_dem_north = #CGIMP DEM TIF file 
year_range = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] #Years we are looking at 
month_range = ['1','2','3','4','5','6','7','8','9','10','11','12']   #Months we are looking at
orbit_range = ['A', 'D'] #Ascending and Descending orbits
location_coord = ["NEEM","Summit","Raven","Similar_NEEM","50km_NEEM"] #SET the LOCATION
mode_list = ['LRM_without_outliers', 'LRM_with_outliers', 'LRM_only_outliers']   #modes with outliers included, seperated, or removed.


#Range Bin to Range Parameters
C, B, Ns_LRM, Ns_SARin, threshold_value = 299792458, 320000000, 128, 1024, 0.05         # Speed of light in m/s,      # Measured chirp bandwidth in Hz,   # Number of LRM Samples,  # Number of SARin Samples,   #was int(0.05) which I think is just 0  
n_SARIN, n_LRM = range(Ns_SARin), range(Ns_LRM)

#Create coordinate dictionary of 3 LRM sites in WGS1984
coordinate_dict = {"50km_NEEM": Point((-49.31994,77.20051)),"NEEM": Point((-51.06,77.45)), "Summit": Point((-38.4500,72.5833)), "Raven": Point((-46.2849,66.4964)), "Similar_NEEM": Point((-40.0,77.6))}
#https://www.vercalendario.info/en/how/convert-latitude-longitude-degrees-decimals.html

#For a wide range of outliers
threshold_index = 0.95

#Outlier Parameters
lrm_start_average = np.array([0, 137.8, 137.6, 363.8, 359.2, 1194])  # Lrm 1, 2, 3, 50, 80,  # I averaged out "waveforms that look nice to me and averaged them out....
lrm_start_mean = np.mean(lrm_start_average)

#Reprojection Parameters
a, b = 6378137, 6356752.3142              #a is the semimajor axis of the WGS84 ellipsoid and b is the semiminor axis of the WGS84 ellipsoid.
e = math.sqrt(1 - ((b**2)/(a**2)))       #e is the eccentricity of the WGS
lat0, lon0 = math.radians(70), math.radians(-45)       # latitude and longitude of the projected origin, and standard parallel/central meridian
mc = (math.cos(lat0) / math.sqrt(1 - ((e**2) * ((math.sin(lat0))**2))))
tc_n, tc_d = math.tan((math.pi / 4) - (lat0 / 2)), ((1 - e * math.sin(lat0)) / (1 + e * math.sin(lat0)))**(e/2)
tc = tc_n / tc_d

#GIMP DEM Polar Stereographic
tif = rt.open(gimp_dem_north)
tif_array = (tif.read(1))
tif_array = np.flipud(tif_array)

#Raster shape and pixel resolution for GIMP DEM Polar Stereographic
pix_res = int(90) #Pixel resolution  (m)
min_x_ext = str(int(-640001)) #Min x extent
max_x_ext = str(int(855799))  #Max x extent
min_y_ext = str(int(-3355551)) #Min y exent
max_y_ext = str(int(-655551))  #Max y exent


epoch_time = datetime(1970, 1, 1)   #Start date counting from 1/1/1970
seconds_week = 604800    #Seconds per week

#Parameter for buffering distance 
distance = 20000  #20km^2, meter radius


#Creation of functions tht will be used 
def location_finder(lon, lat):   #Given lat, lon: locate where it is on a particular raster, in this case GIMP north 
    #Converting from WGS84 to NSIDC Stereographic North (reprojecting)
    #https://stackoverflow.com/questions/42777086/how-to-convert-geospatial-coordinates-dataframe-to-native-x-y-projection
    global min_y_ext, min_x_ext, max_x_ext, pix_res, a, b, e, lat0, lon0, mc, tc
    lon, lat = (math.pi * float(lon/180),  (math.pi * float(lat/180)))
    t_n, t_d = math.tan((math.pi / 4) - (lat / 2)), ((1 - e * math.sin(lat)) / (1 + e * math.sin(lat)))**(e/2)
    t = (t_n / t_d)
    rho = (a * mc * t / tc)
    x_out, y_out = (rho * math.sin(lon - lon0)), ((-1) * rho * math.cos(lon - lon0))
    
    #Creates an array for each corresponding northing and easting with the pixel resolution defined 
    lat_range = np.arange(int(min_y_ext), int(max_y_ext), pix_res)
    lon_range = np.arange(int(min_x_ext),int(max_x_ext),pix_res)
    
    #Calculate the exact x,y position of the station 
    for x in range(len(lon_range)):
        if lon_range[x] > x_out:
            pixel_two = int(x)
            break
    for y in range(len(lat_range)):
        if lat_range[y] > y_out:
            pixel_one = int(y)
            break
    return pixel_one, pixel_two

#Time within Start and end date function
def time_in_range(start, end, x):
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

#Code to pull out LRM and SARin 1B within a specific location and time frame
def waveform_1b_gris(year_interest, month_interest, day, shapefile, orbit):
    global location_finder, tif_array
    startDate = datetime.strptime(("".join([year_interest,month_interest,day])),"%Y%m%d")
    dir_lrm_1b = #directory to SIR_LRM_L1 files
    list_months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12} # List of Months
    
    lrm_array_1b = np.empty([0, 6])
    ####################################################
    ####################################################
    # LRM Mode   
    for file in glob.glob(('/'.join([dir_lrm_1b, year, "{:02d}".format(int(month_interest)), '*.nc']))):   #(dir_lrm_1b + '/' + year + '/' + month + '/' + '*.nc')
        ds = nt.Dataset(file)
        start_time = ds.__dict__['sensing_start']
        orbit_flag = ds.__dict__['ascending_flag']
        
        #pull out date information
        date_information = start_time.split(' ')[0]
        year_2 = date_information.split('-')[2]
        month_2 = date_information.split('-')[1]
        month_3 = list_months[month_2]
        day_2 = date_information.split('-')[0]
        date_2 = ("".join([year_2,str(month_3),day_2]))

        datetime_obj_2 = datetime.strptime(date_2, "%Y%m%d")
    
        
        prod_error = int(ds.__dict__['product_err']) #Added in the product flag.....
        lb_proc_flg = int(ds.__dict__['l1b_proc_flag']) # Added in level 1b processing flag
        l0_proc_flg = int(ds.__dict__['l0_proc_flag']) # Added in level 0 processing flag
        leap_error = int(ds.__dict__['leap_err']) # added in leap error? # timing issue? 
        
        #locating waveforms of the correct orbital path and date, and that there are no errors
        if datetime_obj_2 == startDate and prod_error == 0 and lb_proc_flg == 0 and l0_proc_flg == 0 and leap_error == 0 and orbit_flag == orbit:
            #create arrays with waveform data
            lat, lon, wave, time_delay = np.array(ds['lat_20_ku']), np.array(ds['lon_20_ku']), np.array(ds["pwr_waveform_20_ku"]), np.array(ds["window_del_20_ku"])
            file_name = np.array([file]*len(lat))
            lrm_array_int = [np.array([file_name[i], i, lon[i], lat[i], wave[i], time_delay[i]]) for i in range(len(lat))]
            lrm_array_1b = np.append(lrm_array_1b, lrm_array_int[:], axis = 0)
    #print("Finished with LRM")
    ####################################################
    # Convert  LRM Arrays to Dataframes and Clip based on location of interest
    LRM_1b_dataset = pd.DataFrame({'File Name': lrm_array_1b[:,0], 'Position': lrm_array_1b[:,1], 'Lon': lrm_array_1b[:,2], 'Lat': lrm_array_1b[:,3], 'Wave': lrm_array_1b[:,4], 'Time Delay': lrm_array_1b[:,5]})
    LRM_1b_gdf =  geo.GeoDataFrame(LRM_1b_dataset, geometry=geo.points_from_xy(LRM_1b_dataset.Lon, LRM_1b_dataset.Lat), crs=4326) 
    LRM_1b_clip  = geo.clip(LRM_1b_gdf, shapefile)
    
    return LRM_1b_clip

# https://stackoverflow.com/questions/63040566/how-can-i-show-elements-index-number-from-a-given-list-on-hover-in-matplotlib-g
# function to highlight index of a given plot
def show_annotation(sel):
    ind = int(sel.target.index)
    x, y = sel.target
    sel.annotation.set_text(f' index:{ind}')

# https://stackoverflow.com/questions/41346403/return-index-of-last-non-zero-element-in-list
#function that locates the last non-zero value
def last_non_zero_index(powerArray_to_list):
    # Define convert array input to a list
    temp = powerArray_to_list[:]
    # if the last element list is a zero, delete it
    while temp[-1] == 0:
        del temp[-1]
    # return the last element in the array
    return len(temp) - 1

#functions that locates the last non-zero value of a waveform (v2)
def last_zero_index_better(array):
    zeroes = np.where(np.logical_and(array >= 0, array <= 0.1))
    last_zero = zeroes[0][-1]
    return last_zero

#function that smooths out an array
def smooth(array):
    lrm_smooth = scipy.signal.savgol_filter(array, 29, 2)
    return lrm_smooth

#function that smooths out an array, but slightly less than the "smooth" function
def smooth_low(array):
    lrm_smooth = scipy.signal.savgol_filter(array, 17, 2)
    return lrm_smooth

#function that takes the derivative of an array
def derivative(array):
    deriv = np.diff(array)
    return deriv

#function that finds where the LRM should be clipped off at.
def intro_clip_deriv(array):
    deriv = derivative(smooth(array))
    for i in range(len(deriv)):
        if deriv[i] >= 0:
            clip_intro = i + 1
            break
    return clip_intro

#function that finds how many days of the month there are
def days_in_month(month):
    global year
    if month in days_thirty:
        day_end = int(30)
    if month in days_thirty_one:
        day_end = int(31)
    if int(month) == int(2):
        if year in leap_year:
            day_end = int(29)
        else:
            day_end = int(28)
    return day_end
    
#function that finds Level 1B Metrics. Emphasis on LeW.
def parameters(df):
    df_new = df
    df_new = df_new.reset_index(drop = True)
    LeW_array = np.array([])
    TeS_array = np.array([])
    TeS_outlier_array = np.array([])
    rmse_array = np.array([])
    integrate_array = np.array([])
    for i in range(len(df)):
        waveform = df.iloc[i, 4]
        df_x = waveform[0][:, 0]
        df_y = waveform[0][:, 1]
        smooth_y = smooth_low(df_y) #smooth low is 17, 2
        deriv = derivative(smooth(df_y)) #smooth is 29,2
        range_max = np.where(smooth_y == max(smooth_y))[0][0]
        LeW = df_x[range_max]
        LeW_array = np.append(LeW_array, LeW) #Calculating LeW as range max
        try:    #Calculating the TeS (more complicated)
            flipped_deriv = np.flip(deriv[(range_max+5):])
            last_value_first = np.where(flipped_deriv == max(flipped_deriv))[0][0]
            last_value = len(flipped_deriv) - last_value_first + range_max
            last_list_x, last_list_y = list(df_x[range_max:last_value]), list(df_y[range_max:last_value])
            model = np.poly1d(np.polyfit(last_list_x, last_list_y, 1))
            returned_y = [model(x) for x in last_list_x]
            m = (returned_y[1] - returned_y[0])/(last_list_x[1] - last_list_x[0])
            RMSE = math.sqrt(np.square(np.subtract(last_list_y,returned_y)).mean())     #https://www.askpython.com/python/examples/rmse-root-mean-square-error
            integration = np.sum(df_y)
            TeS_array = np.append(TeS_array, m)
            rmse_array = np.append(rmse_array, RMSE)
            integrate_array = np.append(integrate_array, integration)
            
            for l in range(len(flipped_deriv[:last_value_first])): # range(len(flipped_deriv)-1) : added in the -1 and the l < last_value_first
                if l != 0 and (l < last_value_first) and (flipped_deriv[l - 1] < flipped_deriv[l] > flipped_deriv[l + 1]) and ((threshold_index * flipped_deriv[last_value_first]) < flipped_deriv[l] < ((2.0 - threshold_index) * flipped_deriv[last_value_first])):
                    last_value = len(flipped_deriv) - l + range_max
                    last_list_x, last_list_y = list(df_x[range_max:last_value]), list(df_y[range_max:last_value])
                    model = np.poly1d(np.polyfit(last_list_x, last_list_y, 1))
                    returned_y = [model(x) for x in last_list_x]
                    m = (returned_y[1] - returned_y[0])/(last_list_x[1] - last_list_x[0])
                    RMSE = math.sqrt(np.square(np.subtract(last_list_y,returned_y)).mean())     #https://www.askpython.com/python/examples/rmse-root-mean-square-error
                    integration = np.sum(df_y)
                
                    TeS_array = np.delete(TeS_array, -1)
                    rmse_array = np.delete(rmse_array, -1)
                    integrate_array = np.delete(integrate_array, -1)
                    
                    TeS_array = np.append(TeS_array, m)
                    rmse_array = np.append(rmse_array, RMSE)
                    integrate_array = np.append(integrate_array, integration)
                    break
        except:
            TeS_outlier_array = np.append(TeS_outlier_array, int(i))
            m, RMSE, integration = 0, 0, 0
            TeS_array = np.append(TeS_array, m)
            rmse_array = np.append(rmse_array, RMSE)
            integrate_array = np.append(integrate_array, integration)
            continue
        
    df_new['LeW'] = LeW_array.tolist()
    df_new['TeS'] = TeS_array.tolist()
    df_new['RMSE'] = rmse_array.tolist()
    df_new['Integration'] = integrate_array.tolist()
    
    TeS_error_list = list(TeS_outlier_array)
    outlier_TeS_df = df_new.iloc[TeS_error_list, :]
    
    df_new = df_new.drop(TeS_error_list, axis = 0)
    df_new = df_new.reset_index(drop = True)
    
    return df_new, outlier_TeS_df
    
#Manual Projection from WGS84 to NSIDC3413, Adapted from JWC
def projection(df,a,b,e,lat0,lon0,mc,tc):
    #Converting from WGS84 to NSIDC Stereographic North
    #https://stackoverflow.com/questions/42777086/how-to-convert-geospatial-coordinates-dataframe-to-native-x-y-projection
    df_new = df
    Proj_x = [] #blank array to fill in with projected x coordinates
    Proj_y = [] #blank array to fill in with projected y coordinates
    
    for index, row in df_new.iterrows(): #iterate over rows in the dataframe
        lon, lat = (math.pi * float(row['Lon'])/180),  (math.pi * float(row['Lat'])/180) 
        t_n, t_d = math.tan((math.pi / 4) - (lat / 2)), ((1 - e * math.sin(lat)) / (1 + e * math.sin(lat)))**(e/2)
        t = (t_n / t_d)
        rho = (a * mc * t / tc)
        x_out, y_out = (rho * math.sin(lon - lon0)), ((-1) * rho * math.cos(lon - lon0))
        Proj_x, Proj_y = np.append(Proj_x, x_out), np.append(Proj_y, y_out)

    df_new['Proj_x'] = Proj_x #adding all new x coordinates to a new column in the dataframe
    df_new['Proj_y'] = Proj_y #adding all new y coordinates to a new column in the dataframe
    return df_new

#Function to convert relevant dataframes to CSV and put the name of it in a new row of a .txt file
def csv(df, mode, orbit):
    global lrm_csv_txt, sarin_csv_txt, coord_loc
    csv_file_folder = "\\".join([raw_dir,coord_loc, mode, orbit])                
    if mode == "LRM_lvl_2":
        name_of_file = "_".join([mode, month, day, year])                         
        csv_file_full_path = "/".join([csv_file_folder,("".join([name_of_file,ext]))])                
        df.to_csv(csv_file_full_path, index = False)  
    if mode == "LRM_without_outliers":
        name_of_file = "_".join([mode, month, day, year])                                            
        csv_file_full_path = "/".join([csv_file_folder,("".join([name_of_file,ext]))])                                                                                          
        df.to_csv(csv_file_full_path, index = False) 
    if mode == "LRM_only_outliers":
        name_of_file = "_".join([mode, month, day, year])
        csv_file_full_path = "/".join([csv_file_folder,("".join([name_of_file,ext]))])                
        df.to_csv(csv_file_full_path, index = False)  
    if mode == "LRM_with_outliers":
        name_of_file = "_".join([mode, month, day, year])
        csv_file_full_path = "/".join([csv_file_folder,("".join([name_of_file,ext]))])                
        df.to_csv(csv_file_full_path, index = False) 
    return None

for coord_loc in location_coord:   #Cycle through location
    #Make New Directories with the specific location
    if not exists("\\".join([raw_dir,coord_loc])):
        #makedirs("\\".join([cluster_dir,coord_loc]))
        shutil.copytree(file_structure, "\\".join([raw_dir,coord_loc]))
    
    #Create 20km^2 buffer shapefile around chosen LRM site
    pt_df = geo.GeoDataFrame(geometry=[coordinate_dict[coord_loc]], crs=4326)
    buff = pt_df.copy()
    buff = buff.to_crs(3413)
    buff['geometry'] = buff.geometry.buffer(distance)
    shapefile = buff.to_crs(4326)

    #Finally Starting the Extraction and Parameterization Process!
    for year in year_range:
        for month in month_range:
            #Finding how many days are in the month i'm looking at
            day_total = days_in_month(month)
            
            list_dates_int = list(range(1,(day_total + 1))) #Creates list of int for each day of the month
            day_range = list(map(str, list_dates_int)) #converts that list of int to a list of strings
            
            for day in day_range:
                print(coord_loc,'date:',month,'/',day,'/',year)
                for orbit in orbit_range:
                    #If the particular year/month hasn't been run, then go forward!!!!
                    if not exists("_".join(["\\".join(["\\".join([raw_dir,coord_loc]), 'LRM',f"{orbit}""\LRM"]),month,day,".".join([year, "csv"])])):
                        
                        year_interest, month_interest, day_interest = year, month, day
                        
                        
                        #If a given month doesn't exist in the dataset, skip it!
                        if exists("/".join([lrm_1b_directory,year,"{:02d}".format(int(month_interest))])) is False and exists("/".join([sarin_1b_directory,year,"{:02d}".format(int(month_interest))])) is False:
                            continue
                        
                        # Extracting LRM 1B data
                        lrm_1b = waveform_1b_gris(year_interest, month_interest, day_interest, shapefile, orbit)

                        
                        #Add elevation arrays
                        elevation_array_lrm = np.empty(len(lrm_1b)) 
                      
                        if len(lrm_1b) != 0:
                            #Add in Elevation
                            for i in range(len(lrm_1b)):
                                location = location_finder(lrm_1b.iloc[i]['Lon'], lrm_1b.iloc[i]['Lat'])
                                pixel_1, pixel_2 = location[0], location[1]
                                elevation_array_lrm[i] = tif_array[pixel_1,pixel_2]
                            elevation_list_lrm = elevation_array_lrm.tolist()
                            lrm_1b.insert(5, "Elevation", elevation_list_lrm, True)
                                
                            #Converting LRM range bin to range, and then centering the waveform around the threshold of where the signal ramps up
                            lrm_threshold_error_array = np.array([])
                            for i in range(len(lrm_1b)):
                                Tw_LRM = lrm_1b.iloc[i, 6]  #time delay
                                range_LRM = [(C/(int(2)*B))*((Tw_LRM*B)-(Ns_LRM/int(2)) + n_LRM.index(wave)) for wave in range(Ns_LRM)] #Converting to Range from range bin
                            
                                power_LRM = np.array((lrm_1b.iloc[i, 4]))  # Waveform power values themselves
                                max_index = np.where(power_LRM == max(power_LRM))[0][0] # the max index is where the power is the highest, if there is more than one it is the first one.
                                # the average first power is the average of the 10th to 16th (because the first 10 are the leftovers of the previous waveform)
                                range_to_clip = intro_clip_deriv(power_LRM)
                                first_power_lrm = power_LRM[range_to_clip:(range_to_clip + 10)]
                                average_first_power = np.mean(first_power_lrm)
                                # define the minimum index as the last minimum of the power_LRM
                                min_index = np.where(power_LRM == min(power_LRM))[0][-1]
                                clipped_array = power_LRM[range_to_clip:]
                                #dont clip if the waveform leading edge is more than the 64th bin, and skip
                                if min_index > (128/2): 
                                    threshold_index = 0
                                # if the minimum index is 0
                                if (clipped_array[min_index-range_to_clip] >= 0) and (clipped_array[min_index-range_to_clip] < 0.1):
                                    #locate last zero value of the waveform
                                    last_zero = last_zero_index_better(clipped_array)
                                    #clip array right before the last zero.
                                    new_array = clipped_array[(last_zero - 1):]
                                    if len(new_array) != 0:
                                        for x in range(len(new_array)):
                                            try:
                                                # cycle through new array and if the power value is greater than [the mean of the beginning of the clipped array + two standard deviation of the beginning] and that the array i+1 is greater than i:
                                                if new_array[x] > ((np.mean(clipped_array[0:(last_zero)])) + (2*np.std(clipped_array[0:(last_zero)]))) and (new_array[x+1] > new_array[x]):
                                                    threshold_index = int(x) + int(range_to_clip) + int(last_zero) - 1
                                                    break
                                            except:
                                                threshold_index = 0
                                                break
                                    else:
                                        #Add in case previous step removes what was done.
                                        threshold_index = 0
                                    if threshold_index > (128/2):
                                        threshold_index = 0
                                else:
                                    try:
                                        new_array = clipped_array[(min_index-1):] #this time make the new array based off the min index rather than the last zero.
                                        if len(new_array) != 0:
                                            for x in range(len(new_array)):
                                                if new_array[x] > ((np.mean(clipped_array[0:(min_index)])) + (2*np.std(clipped_array[0:(min_index)]))) and (new_array[x+1] > new_array[x]):
                                                    threshold_index = int(x) + int(range_to_clip) + int(min_index) - 1
                                                    break
                                        else:
                                            threshold_index = 0
                                        if threshold_index > (128/2):
                                            threshold_index = 0
                                    except IndexError:
                                        threshold_index = 0
                                centered_range_array = np.empty([(Ns_LRM)])
                                range_of_threshold = range_LRM[threshold_index]
                                
                                #delegate arrays with a threshold index of 0 to the error / outlier list.
                                if threshold_index == 0:
                                    lrm_threshold_error_array = np.append(lrm_threshold_error_array, i)
                                
                                #center the waveform using the threshold index
                                for l in range(128):  
                                    new_range_value = range_LRM[l] - range_of_threshold
                                    centered_range_array[l] = new_range_value
                                centered_waveform = np.delete(((np.vstack([centered_range_array, power_LRM])).T), np.s_[0:range_to_clip], axis=0)
                                lrm_1b.iloc[i, 4] = [centered_waveform]
                            
                            #Dropping LRM waveforms that don't meet threshold criteria, and making a LRM threshold DF
                            lrm_1b_outlier_threshold_df = lrm_1b.iloc[list(lrm_threshold_error_array), :]
                            lrm_1b = lrm_1b.reset_index(drop=True)
                            lrm_1b = lrm_1b.drop(list(lrm_threshold_error_array), axis=0)   #was a list of a list
                    
                            #Removing other LRM outliers based on values and derivative
                            
                            #https://www.geeksforgeeks.org/python-check-if-all-the-values-in-a-list-are-less-than-a-given-value/#:~:text=Using%20all()%20function%20we,values%2C%20else%20it%20returns%20false.
                            lrm_outlier_array = np.array([])
                            for i in range(len(lrm_1b)):
                                LRM_waveform_tuple = lrm_1b.iloc[i, 4]
                                lrm_x = LRM_waveform_tuple[0][:, 0]   # lrm_x is the range
                                lrm_y = LRM_waveform_tuple[0][:, 1] # lrm_y is the power
                                lrm_noise_average = np.mean(lrm_y[0:9]) # the noise average is the average of the first ten of them (DID I INDEX THEM CORRECTLY?????????)
                                try:
                                    range_zero = int(np.where(lrm_x == 0)[0]) # calculate the range of zero as where the range = 0
                                except TypeError:
                                    lrm_outlier_array = np.append(lrm_outlier_array, i) # if there is an error in this, put it in the outlier list
                                    continue
                                #lrm_b4_zero = lrm_y[(range_zero-8):range_zero-2] 
                                lrm_b4_zero = np.mean(lrm_y[(range_zero-8):range_zero-2], axis=0) # average the lrm_b4_zero,  the lrm before zero is the power from range_zero -8 to range_zero-2
                                lrm_y_smooth = scipy.signal.savgol_filter(lrm_y, 19, 2) # smooth out the lrm power using the savgol filter to remove the noise, and just keep the overall pattern
                                lrm_y_deriv = np.diff(lrm_y_smooth) # find the derivative of the smoothed out line....
                                range_max = int(np.where(lrm_y_deriv == max(lrm_y_deriv))[0][0]) # find the maximum deriviatve value, which is generally where the maximum point is......
                                deriv_array_test = lrm_y_deriv[range_zero:range_max] # define the point of interest to be between the threshold to the maximum or the peak.... (might want to change the max_range to a literal max rather than a maximum of the derivative?)
                                # https://stackoverflow.com/questions/65791345/problems-with-numpy-any-for-condition-checking
                                if (lrm_b4_zero >= (lrm_start_mean + (2*np.std(lrm_start_average)))) or (any(x < 500 for x in deriv_array_test)):  # if the lrm_b4_zero is greater than or the standard deviation of the rise OR the derivative is less than 500 between the threshold and the maximum (need to take a look at this), make it an outlier!
                                    lrm_outlier_array = np.append(lrm_outlier_array, int(i))
                           
                            #removing LRM deriv and value outliers and making it its own DF
                            lrm_1b_outlier_value_deriv_df = lrm_1b.iloc[list(lrm_outlier_array), :]
                            lrm_1b = lrm_1b.reset_index(drop=True)
                            lrm_1b = lrm_1b.drop(list(lrm_outlier_array), axis=0)
                        
                            #Parameterizing LRM Waveforms (Pulling out TeS, RMSE of TeS, LeW, Sigma Alex)
                            results_parameters_LRM = parameters(lrm_1b)
                            lrm_1b = results_parameters_LRM[0]
                            lrm_1b_outlier_TeS_df = results_parameters_LRM[1]
                
                            #Combining the LRM Outliers to Create ONE dataframe
                            lrm_total_outliers_df = pd.concat([lrm_1b_outlier_threshold_df,lrm_1b_outlier_value_deriv_df,lrm_1b_outlier_TeS_df])
                            
                            #Reprojecting from WGS1984 to NSIDC Stereographic North
                            lrm_total_outliers_df = projection(lrm_total_outliers_df,a,b,e,lat0,lon0,mc,tc)
                            lrm_1b  = projection(lrm_1b,a,b,e,lat0,lon0,mc,tc)
                            
                            
                            #Add outlier identification tag
                            lrm_total_outliers_df['Outlier'] = (['Yes']* len(lrm_total_outliers_df))
                            lrm_1b['Outlier'] = (['No'] * len(lrm_1b))
                            
                            #Add Orbit Information
                            orbit_info_no =  [orbit] * len(lrm_1b)
                            lrm_1b['Orbit'] = orbit_info_no

                            orbit_info_yes =  [orbit] * len(lrm_total_outliers_df)
                            lrm_total_outliers_df['Orbit'] = orbit_info_yes
                            
                            #Reintroduce Outliers
                            lrm_1b_with_outliers = pd.concat([lrm_1b,lrm_total_outliers_df])
                            
                            #Save files!
                            csv(lrm_total_outliers_df, 'LRM_only_outliers', orbit)
                            csv(lrm_1b, "LRM_without_outliers", orbit)
                            csv(lrm_1b_with_outliers, "LRM_with_outliers", orbit)
                            
                            
    #Clustering/averaging the data
    second_to_date = []
    df_total = pd.DataFrame(columns=['File Name', 'Position', 'date', 'Time Delay', 'TeS','LeW','Integration','RMSE', 'Lat', 'Lon', 'Elevation', 'Orbit', 'Outlier'])
    for mode in mode_list:
        df_total = pd.DataFrame(columns=['File Name', 'Position', 'date', 'Time Delay', 'TeS','LeW','Integration','RMSE', 'Lat', 'Lon', 'Elevation', 'Orbit', 'Outlier'])
        for orbit in orbit_range:
            for file in listdir("\\".join(["\\".join([raw_dir,coord_loc]), mode, orbit])):
                date_txt = file.split('\\')[-1][:-4]
                year, month, day = date_txt.split('_')[-1],   "{:02d}".format(int(date_txt.split('_')[-3])),  "{:02d}".format(int(date_txt.split('_')[-2]))
                #date_join = (":".join([year,str(month),day]))
                datetime_obj = datetime(int(year), int(month), int(day), 0, 0)
                #datetime_obj = datetime.strptime(date_join, "%Y%d%m")
                delta = (datetime_obj - epoch_time)
                delta = delta.total_seconds()
                df = pd.read_csv("\\".join(["\\".join([raw_dir,coord_loc]),mode,orbit,file]))
                date_list = [delta] * len(df)
                lew, tes, rmse, integration, lat, lon = df["LeW"], df["TeS"], df["RMSE"], df["Integration"], df["Lat"], df['Lon']
                elevation, orbit_info, outlier, position, filename, timedelay = df['Elevation'], df['Orbit'], df['Outlier'], df['Position'], df['File Name'], df['Time Delay']
                temp_df = pd.DataFrame(columns=['date','TeS','LeW', 'Integration','RMSE', 'Lat', 'Lon', 'Elevation','Orbit','Outlier', 'Time Delay', 'File Name', 'Position'])
                temp_df["date"], temp_df["TeS"], temp_df['LeW'], temp_df['Integration'] = date_list, tes.tolist(), lew.tolist(), integration.tolist()
                temp_df['RMSE'], temp_df['Lat'], temp_df['Lon'], temp_df['Elevation'] = rmse.tolist(), lat.tolist(), lon.tolist(), elevation.tolist()
                temp_df['Orbit'], temp_df['Outlier'], temp_df['Position'], temp_df['File Name'], temp_df['Time Delay'] = orbit_info.tolist(), outlier.tolist(), position.tolist(), filename.tolist(), timedelay.tolist()
                df_total = df_total.append(temp_df, ignore_index = True) #creating the entire dataframe of relevant waveforms, locations, etc. 
        df_total = df_total.sort_values(by='date')
        df_total = df_total.reset_index(drop=True)
    
        df_total.to_csv("\\".join([totals_dir,"".join([f"total_{coord_loc}_{mode}",ext])])) #convert DF to csv file
        if mode == "LRM_without_outliers":   #Do not include the errors / outliers when clustering
            
            #create a blank df that will be filled in as the clustering goes on.                      
            df_average = pd.DataFrame(columns=['date', 'TeS','LeW','Integration','RMSE', 'Lat', 'Lon', 'Elevation',"#Samples","Std-Int","Std-LeW","Std-TeS","Std-RMSE"])  #Added in # of samples, and STD values for each metric
        
            row_list = []
            for i in range(len(df_total)):
                row_list.append(i)
                first_date = row_list[0]
                try:
                    if ((df_total.loc[i, "date"] + seconds_week) >= df_total.loc[i+1,"date"]): #The 1st through N-1 data in the weekly time frame is included
                        print("normal", i)
                        continue
                    df_temp_average = df_total.iloc[row_list,:].mean(axis=0)
                    std_Integration =  df_total.loc[row_list,'Integration'].std()  #STD for different metrics
                    std_LeW = df_total.loc[row_list,'LeW'].std()
                    std_TeS = df_total.loc[row_list,'TeS'].std()
                    std_RMSE = df_total.loc[row_list,'RMSE'].std()
                    number_population = len(row_list)         #Calculates number of total samples that are being averaged.
                    df_temp_average['#Samples'] = number_population        #Adding in STD values to a temporary average df which will then be appended onto the df_average.
                    df_temp_average['Std-Int'] = std_Integration
                    df_temp_average['Std-LeW'] = std_LeW
                    df_temp_average['Std-TeS'] = std_TeS
                    df_temp_average['Std-RMSE'] = std_RMSE
                    df_average = df_average.append(df_temp_average, ignore_index = True)
                    row_list = []
                    continue
                except:
                    if df_total.loc[i-1,"date"] + seconds_week >= df_total.loc[i, "date"]: #The last date in the weekly time frame is included
                        print("last normal", i)
                        df_temp_average = df_total.iloc[row_list,:].mean(axis=0)
                        std_Integration = df_total.loc[row_list,'Integration'].std()
                        std_LeW = df_total.loc[row_list,'LeW'].std()
                        std_TeS = df_total.loc[row_list,'TeS'].std()
                        std_RMSE = df_total.loc[row_list,'RMSE'].std()
                        number_population = len(row_list)
                        df_temp_average['#Samples'] = number_population
                        df_temp_average['Std-Int'] = std_Integration
                        df_temp_average['Std-LeW'] = std_LeW
                        df_temp_average['Std-TeS'] = std_TeS
                        df_temp_average['Std-RMSE'] = std_RMSE
                        
                        df_average = df_average.append(df_temp_average, ignore_index = True)
                    else:                                   #Last one in the time series total.
                        print("last not normal", i)                                      
                        df_temp_average = df_total.iloc[row_list[:-1],:].mean(axis=0)
                        std_Integration = df_total.loc[row_list[:-1],'Integration'].std()
                        std_LeW = df_total.loc[row_list[:-1],'LeW'].std()
                        std_TeS = df_total.loc[row_list[:-1],'TeS'].std()
                        std_RMSE = df_total.loc[row_list[:-1],'RMSE'].std()
                        number_population = len(row_list[:-1])
                        df_temp_average['#Samples'] = number_population
                        df_temp_average['Std-Int'] = std_Integration
                        df_temp_average['Std-LeW'] = std_LeW
                        df_temp_average['Std-TeS'] = std_TeS
                        df_temp_average['Std-RMSE'] = std_RMSE
                        df_average = df_average.append(df_temp_average, ignore_index = True)
                        df_average = df_average.append(df_total.iloc[i])
                  
            #Make date in appropriate type.        
            list_seconds = df_average['date'].tolist()
            [second_to_date.append(datetime.fromtimestamp(int(seconds)).strftime('%m-%d-%Y')) for seconds in list_seconds]
            df_average['date'] = second_to_date
        
            avg_lat, avg_lon = df_average['Lat'].tolist(), df_average['Lon'].tolist()
            elevation_avg_coord_list = []
        
            for lat, lon in zip(avg_lat, avg_lon):    #add in elevations derived from GIMP DEM
                location = location_finder(lon, lat)
                pixel_1, pixel_2 = location[0], location[1]
                elevation_avg_coord = tif_array[pixel_1,pixel_2]
                elevation_avg_coord_list.append(elevation_avg_coord)
        
            df_average["elevation of avg coord"] = elevation_avg_coord_list
            df_average.to_csv("\\".join([cluster_dir,"".join([f"cluster_{coord_loc}_{mode}",ext])])) #Save clustered level 1B metrics as a CSV file for later use. 
            second_to_date = []                                                
    #     #######################################################################################################
          
