# -*- coding: utf-8 -*-
#Importing Modules
import os
import datetime
from ftplib import FTP

#Connecting to FTP
HOSTNAME = 'science-pds.cryosat.esa.int'
USERNAME = ''
PASSWORD = ''

# Open FTP server.
ftp = FTP(HOSTNAME)
#dest_dir = ""
dest_dir = ""

#Iniate login to FTP server
ftp.login(USERNAME,PASSWORD)


#Importing Target Region for ALL of Greenland
target_region 	= [59, 83, -78, -9]

#Define Directory 
dir = ""


#We are looking for .HDR files that correspond to .NC files
header_ext		= ".HDR"
data_ext		= ".nc"
instrument_mode = "SIR_LRM_L2"

years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']


#Where I want the final list of NC files to go
#Creates a new "Output file everytime I run the code with a different time stamp.

dest_folder = 

#(" ", "_")

#Modified from Bob Hawley

#Creating a blank array for future use
found_files = []

#Look through all files that meet criteria and download those that meet criteria.
 #Years for loop
for year in years:        
                        
    months = os.listdir(dir + '/' + year) 
                                    
    for month in months: 
        #opening months                                
        files = os.listdir(dir + '/' + year + '/' + month)  
        #opening the file                           
        for file in files:  
            #defining HDR files                                     
            if file[-4:] == header_ext:                                
                #Opening the files
                header = open(dir + '/' + year + '/' + month + '/' + file, 'r')
                #Initializing value that wouldn't work.
                lat = 500
                lon = 500
                
                for line in header.readlines():
                    location = str.find(line, "</Start_Lat>")
                    if location != -1:
                        lat = float(line[location-11:location-6] + "." + line[location-5:location])
                        #print(lat)
                    location = str.find(line, "</Start_Long>")
                    if location != -1:
                        long = float(line[location-11:location-6] + "." + line[location-5:location])
                        #print(long)
                    location = str.find(line, "</Stop_Lat>")
                    if location != -1:
                        stop_lat = float(line[location-11:location-6] + "." + line[location-5:location])
                        #print(stop_lat)
                    location = str.find(line, "</Stop_Long>")
                    if location != -1:
                        stop_long = float(line[location-11:location-6] + "." + line[location-5:location])
                        #print(stop_long)
                header.close()
                
                #Saves HDR files within the bounding box into the blank array
                if ((lat + stop_lat)/2 >= target_region[0]) & ((lat + stop_lat)/2 <= target_region[1]) & ((long + stop_long)/2 >= target_region[2]) & ((long + stop_long)/2 <= target_region[3]):
                    nc_file = file[:-4] + data_ext
                    if not os.path.exists(dir + '/' + year + '/' + month + '/' + nc_file):
                        found_files.append((dir + '/' + year + '/' + month + '/' + file[:-4])) # might want to get rid of eventually.
                        print(dir + '/' + year + '/' + month + '/' + file[:-4] + data_ext)
                        path_name = '/Ice_Baseline_D/' + instrument_mode + '/' + year + '/' + month + '/'
                        ftp.cwd(path_name)
                        path_name_local = dir + '/' + year + '/' + month + '/' + nc_file
                        with open(path_name_local, 'wb') as function:
                            def callback(data):
                                function.write(data)
                            try:
                                ftp.retrbinary('RETR %s' % nc_file, callback, 1024)
                            except OSError or AttributeError or SyntaxError or TypeError:
                                pass
ftp.close()
                    












        
        

    		
             

             









