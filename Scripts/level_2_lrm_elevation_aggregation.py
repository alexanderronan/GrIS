"""
Created on Thu Jan 26 14:06:51 2023

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
lrm_2_dir = #directory of where to find the level 2 lrm data.
ext = '.csv' #CSV extension
year_range = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] #Years we are looking at 
month_range = ['1','2','3','4','5','6','7','8','9','10','11','12']   #Months we are looking at

orbit_range = ['A', 'D'] #Ascending and Descending orbits
location_coord = ["NEEM","Summit","Raven","Similar_NEEM","50km_NEEM"] #SET the LOCATION
#location_coord = ["Summit"]
mode = "LRM" #Where the locations are located

raw_dir = #directory to input intermediary data.
totals_dir = #directory to put final csv file. 


LRM_file_structure = #example file structure to follow: LRM -> [A], [D]

#Parameters for clustering and averaging information
epoch_time = datetime(1970, 1, 1)
seconds_week = 604800


#Buffer distance
distance = 20000
#Reprojection Parameters
a, b = 6378137, 6356752.3142              #a is the semimajor axis of the WGS84 ellipsoid and b is the semiminor axis of the WGS84 ellipsoid.
e = math.sqrt(1 - ((b**2)/(a**2)))       #e is the eccentricity of the WGS
lat0, lon0 = math.radians(70), math.radians(-45)       # latitude and longitude of the projected origin, and standard parallel/central meridian
mc = (math.cos(lat0) / math.sqrt(1 - ((e**2) * ((math.sin(lat0))**2))))
tc_n, tc_d = math.tan((math.pi / 4) - (lat0 / 2)), ((1 - e * math.sin(lat0)) / (1 + e * math.sin(lat0)))**(e/2)
tc = tc_n / tc_d

#Create coordinate dictionary of 3 LRM sites in WGS1984
coordinate_dict = {"50km_NEEM": Point((-49.31994,77.20051)),"NEEM": Point((-51.06,77.45)), "Summit": Point((-38.4500,72.5833)), "Raven": Point((-46.2849,66.4964)), "Similar_NEEM": Point((-40.0,77.6))}
#https://www.vercalendario.info/en/how/convert-latitude-longitude-degrees-decimals.html


#Locating suitable Level 2 elevation data for a given lcoation / time.
def elevation(year_interest, month_interest, day, shapefile, orbit):
    startDate = datetime.strptime(("".join([year_interest,month_interest,day])),"%Y%m%d")
    lrm_array_2  = np.empty([0, 5])
    list_months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12} # List of Months
    for file in glob.glob(('/'.join([lrm_2_dir, year, "{:02d}".format(int(month_interest)), '*.nc']))):   #(dir_lrm_1b + '/' + year + '/' + month + '/' + '*.nc')
        ds = nt.Dataset(file)
        start_time = ds.__dict__['sensing_start']
        orbit_flag = ds.__dict__['ascending_flag']
        
        date_information = start_time.split(' ')[0]
        year_2 = date_information.split('-')[2]
        month_2 = date_information.split('-')[1]
        month_3 = list_months[month_2]
        day_2 = date_information.split('-')[0]
        date_2 = ("".join([year_2,str(month_3),day_2]))
    
        datetime_obj_2 = datetime.strptime(date_2, "%Y%m%d")

        prod_error = int(ds.__dict__['product_err']) #Added in the product flag.....
        leap_error = int(ds.__dict__['leap_err']) # added in leap error? # timing issue? 
        
        if datetime_obj_2 == startDate and prod_error == 0 and leap_error == 0 and orbit_flag == orbit:
            
            lat, lon, elevation_1, elevation_2, elevation_3 = np.array(ds['lat_poca_20_ku']), np.array(ds['lon_poca_20_ku']), np.array(ds['height_1_20_ku']), np.array(ds['height_2_20_ku']), np.array(ds['height_3_20_ku'])
            lrm_array_int = [np.array([lon[i], lat[i], elevation_1[i], elevation_2[i],elevation_3[i]]) for i in range(len(elevation_1))]
            lrm_array_2 = np.append(lrm_array_2, lrm_array_int[:], axis = 0)
            
            
    #print("Finished with LRM")
    ####################################################
    # Convert SARIN and LRM Arrays to Dataframes and Clip
    LRM_2_dataset = pd.DataFrame({'Lon': lrm_array_2[:,0], 'Lat': lrm_array_2[:,1], 'Elevation 1': lrm_array_2[:,2], 'Elevation 2': lrm_array_2[:,3], 'Elevation 3': lrm_array_2[:,4]})
    LRM_2_gdf =  geo.GeoDataFrame(LRM_2_dataset, geometry=geo.points_from_xy(LRM_2_dataset.Lon, LRM_2_dataset.Lat), crs=4326) 
    LRM_2_clip  = geo.clip(LRM_2_gdf, shapefile)

    return LRM_2_clip

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

#Saving as a CSV file in the appropriate areas.
def csv(df, mode, orbit):
    global coord_loc
    csv_file_folder = "\\".join([raw_dir,coord_loc, orbit])                
    if mode == "LRM" and len(df)!= 0:
        name_of_file = "_".join([mode, month, day, year])                         
        csv_file_full_path = "/".join([csv_file_folder,("".join([name_of_file,ext]))])                
        df.to_csv(csv_file_full_path, index = False)  
    return None

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

for coord_loc in location_coord:   #Cycle through location
    #print(coord_loc)
    #Make New Directories with the specific location
    if not exists("\\".join([raw_dir,coord_loc])):
        #makedirs("\\".join([cluster_dir,coord_loc]))
        shutil.copytree(LRM_file_structure, "\\".join([raw_dir,coord_loc]))
    
    #Create 20km buffer shapefile around chosen LRM site
    pt_df = geo.GeoDataFrame(geometry=[coordinate_dict[coord_loc]], crs=4326)
    buff = pt_df.copy()
    buff = buff.to_crs(3413)
    buff['geometry'] = buff.geometry.buffer(distance)
    shapefile = buff.to_crs(4326)
    
    #Start the Extraction and Parameterization Process!
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
                    if not exists("_".join(["\\".join(["\\".join([raw_dir,coord_loc]), mode,f"{orbit}""\LRM"]),month,day,".".join([year, "csv"])])):
                        year_interest, month_interest, day_interest = year, month, day
                        #If a given month doesn't exist in the dataset, skip it!
                        if exists("/".join([lrm_2_dir,year,"{:02d}".format(int(month_interest))])) is False:
                            continue
                        
                        # Extracting SARin and LRM 1B data
                        lrm_2 = elevation(year_interest, month_interest, day, shapefile, orbit)
                        
                        #Adding in projection
                        lrm_2  = projection(lrm_2,a,b,e,lat0,lon0,mc,tc)
                        
                        #Add in orbit information
                        orbit_info =  [orbit] * len(lrm_2)
                        lrm_2['Orbit'] = orbit_info
                        
                        #remove geometries!
                        lrm_2 = lrm_2.drop('geometry', axis=1)
                        #Put to CSV file
                        csv(lrm_2, "LRM", orbit)
                        
                        
    #Clustering/averaging the data
    second_to_date = []
    
    #activating the dataframe that holds everything together.
    df_total = pd.DataFrame(columns=['date', 'Elevation 1','Elevation 2','Elevation 3','Lat', 'Lon','Proj_x','Proj_y','Orbit'])
    #df_average = pd.DataFrame(columns=['date', 'Elevation 1','Elevation 2','Elevation 3','Lat', 'Lon',"#Samples","Std-Elevation1","Std-Elevation2","Std-Elevation3"])  #Added in # of samples, and STD values for each metric

    for orbit in orbit_range:
        for file in listdir("\\".join(["\\".join([raw_dir,coord_loc]), orbit])):
            date_txt = file[4:-4]
            #print(date_txt)
            year, month, day = date_txt.split('_')[2],   "{:02d}".format(int(date_txt.split('_')[1])),  "{:02d}".format(int(date_txt.split('_')[0]))
            date_join = ("".join([year,str(month),day]))
            #print(date_join)
            datetime_obj = datetime.strptime(date_join, "%Y%d%m")
            #print(datetime_obj)
            delta = (datetime_obj - epoch_time)
            delta = delta.total_seconds()
            df = pd.read_csv("\\".join(["\\".join([raw_dir,coord_loc]),orbit,file]))
            date_list = [delta] * len(df)
            elevation_1, elevation_2, elevation_3,lat,lon, orbit_info, Proj_x, Proj_y= df["Elevation 1"], df["Elevation 2"], df["Elevation 3"], df["Lat"], df['Lon'], df['Orbit'], df['Proj_x'], df['Proj_y']
            temp_df = pd.DataFrame(columns=['date','Elevation 1','Elevation 2', 'Elevation 3','Lat', 'Lon', 'Orbit', 'Proj_x', 'Proj_y'])
            temp_df["date"], temp_df["Elevation 1"], temp_df['Elevation 2'], temp_df['Elevation 3'], temp_df['Lat'], temp_df['Lon'], temp_df['Orbit'], temp_df['Proj_x'], temp_df['Proj_y'] = date_list, elevation_1.tolist(), elevation_2.tolist(), elevation_3.tolist(), lat.tolist(), lon.tolist(), orbit_info.tolist(), Proj_x.tolist(), Proj_y.tolist()
            df_total = df_total.append(temp_df, ignore_index = True)
    df_total = df_total.sort_values(by='date')
    df_total = df_total.reset_index(drop=True)

    #Saves the file 
    df_total.to_csv("\\".join(["\\".join([raw_dir,coord_loc]),"".join(["total_v3",ext])]))                                                
   
