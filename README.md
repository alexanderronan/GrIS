# GrIS
Data Repository for "CryoSat-2 Parameterization Across the GrIS."
This repository contains the following paths: [Data](/Data/), [Environments](/Environments/), and [Scripts](/Scripts/).
[Data](/Data/) contains all relevant data files that are generated from the python scripts in [Scripts](/Scripts/). [Environments](/Environments/) contains the python environments needed to run the relevant python scripts. 

--------------------------------------------------------------------------------------------------
[Data](/Data/) contains the following:
  1. [Correlation Analysis](/Data/Correlation_Analysis/) : 
  2. [Graphs](/Data/Graphs/)
  3. [Level 1B LeW](/Data/Level_1B_LeW/)
     * [Raw Data](/Data/Level_1B_LeW/Level_1B_Raw_Data/)
     * [Clustered Data](/Data/Level_1B_LeW/Level_1B_Clustered_Data/)
     * [BEAST Outputs](/Data/Level_1B_LeW/Level_1B_BEAST_Outputs/)
  4. [Level 2 Elevations](/Data/Level_2_Elevations/)
     * [Raw Data](/Data/Level_2_Elevations/Level_2_Elevation_Raw_Data/)
     * [Clustered Data](/Data/Level_2_Elevations/Level_2_Elevation_Clustered_Data/)
     * [Smoothed Data](/Data/Level_2_Elevations/Level_2_Elevation_Lowess/)
     * [BEAST Outputs](/Data/Level_2_Elevations/Level_2_Elevation_BEAST_Outputs/)



[Correlation Analysis](/Data/Correlation_Analysis/) contains the correlation coefficient analysis at Summit and Raven between LeW and Level 2 Elevations.

[Graphs](/Data/Graphs/) Contains the relevant output graphs for each study location.

[Level 1B LeW](/Data/Level_1B_LeW/) contains all Level 1B metrics used in the study. 2A contains the non-clustered level-1b metrics that have been calculated using python scripts for each study site. 2A contains multiple CSV for each files with different either/both erroneous waveform information and calculate metrics. 2B contains the clustered level 1B metrics derived from the non-outliers CSV file in 2A. 2C contains the BEAST algorithm outputs for each site and multiple aggregation periods as a .pkl file.

[Level 2 Elevations](/Data/Level_2_Elevations/) contains all Level 2 retracked elevation data used in the study. 4A contains the raw elevation data in a CSV file for each study location. 4B contains the clustered elevation data in a CSV file for each study location. 4C includes the smoothed profile of the clustered elevation data as CSV files. 1/8th and 1/12th refer to the aggregation period for smoothing. 4D contains the BEAST algorithm outputs for each site and multiple aggregation periods as a .pkl file. 


--------------------------------------------------------------------------------------------------

[Scripts](/Scripts/) contains the following scripts:
  1. [Level 2 Elevation Change Detection](/Scripts/elevation_change_detection.py)
  2. [Level 2 Elevation Lowess](/Scripts/elevation_lowess.py)
  3. [Level 2 Elevation Multiple Regression](/Scripts/elevation_multiple_regression.py)
  4. [Level 2 LRM Elevation Aggregation](/Scripts/level_2_lrm_elevation_aggregation.py)
  5. [Level 1B Metric Change Detection](/Scripts/metric_change_detection.py)
  6. [Level 1B Metric Parameterization and Clustering](/Scripts/parameterization_and_clustering_metrics.py)


[Level 1B Metric Parameterization and Clustering](/Scripts/parameterization_and_clustering_metrics.py) takes Level 1B waveform power values around a given buffer of a given study location and calculates waveform metrics (LeW, TeS, Integration), and clusters the data by calendar week. Erroneous and problematic waveforms are removed. 

[Level 2 LRM Elevation Aggregation](/Scripts/level_2_lrm_elevation_aggregation.py) takes the Level 2 elevation data for a given location and merges them into one .csv file.

[Level 2 Elevation Lowess](/Scripts/elevation_lowess.py) takes Level 2 retracked elevations around a given buffer around the study site and aggregates them on a monthly basis, applying a linear regression to evaluate the elevation at the absolute study site.

[Level 1B Metric Change Detection](/Scripts/metric_change_detection.py) Takes the evaluated elevations from Script #3 and applies a Lowess smoothing function.

[Level 2 Elevation Change Detection](/Scripts/elevation_change_detection.py) Takes the clustered Level 1B metrics and applies the BEAST algorithm to it.

[Level 2 Elevation Change Detection](/Scripts/elevation_change_detection.py) Takes the evaluated Level 2 elevations and applies the BEAST algorithm to it.

Script #7 Plots the Level 2 OCOG & UCL-Land Ice Retracked Elevations for a given study site, as well as its Level 1B LeW over time. 

Non-BEAST algorithms & the plotting algorithm require the included
The remaining algorithms require the included 









