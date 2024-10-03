Radial_profile_analysis_numba.py is the Python script used to estimate the apparent outer rim diameter of a hypervelocity impact crater. It uses digital elevation models (DEMs) to identify topographic rings surrounding an impact structure through an iterative process and assigns the most consistently identified positive relief ring as the putative outer rim.

The script requires a DEM, Easting and Northing (UTM Coordinates) of the crater center, an estimate of the inner ring diameter.

Default options are to run using: '-t complex', '-a 1', '-pc 4', and 'sd 1'.
