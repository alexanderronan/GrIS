# GrIS
Data Repository for "CryoSat-2 Parameterization Across the GrIS."
This repository contains the following paths: [Data](/Data/), [Environments](/Environments/), and [Scripts](/Scripts/).
[Data](/Data/) contains all relevant data files that are generated from the python scripts in [Scripts](/Scripts/). [Environments](/Environments/) contains the python environments needed to run the relevant python scripts. 

--------------------------------------------------------------------------------------------------
[Data](/Data/) contains the following:
  1. [Correlation Analysis](/Data/Correlation_Analysis/)
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



[Correlation Analysis](/Data/Correlation_Analysis/) contains the correlation coefficient analysis at Summit and Raven between LeW and Level 2 Elevations.}

[Graphs](/Data/Graphs/) Contains the relevant output graphs for each study location.\

[Level 1B LeW](/Data/Level_1B_LeW/) contains all Level 1B metrics used in the study. 2A contains the non-clustered level-1b metrics that have been calculated using python scripts for each study site. 2A contains multiple CSV for each files with different either/both erroneous waveform information and calculate metrics. 2B contains the clustered level 1B metrics derived from the non-outliers CSV file in 2A. 2C contains the BEAST algorithm outputs for each site and multiple aggregation periods as a .pkl file.\

[Level 2 Elevations](/Data/Level_2_Elevations/) contains all Level 2 retracked elevation data used in the study. 1A contains the raw elevation data in a CSV file for each study location. 1B contains the clustered elevation data in a CSV file for each study location. 1C includes the smoothed profile of the clustered elevation data as CSV files. 1/8th and 1/12th refer to the aggregation period for smoothing. 1D contains the BEAST algorithm outputs for each site and multiple aggregation periods as a .pkl file. 

--------------------------------------------------------------------------------------------------
