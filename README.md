[[DOI]](https://doi.org/10.5281/zenodo.23503865)

# Radial Profile Analysis System
RadialProfileAnalysisSystem.py is the Python script developed for the paper: [Identifying the Dimensions of Degraded Impact Structures Using Digital Elevation Models: A Proof of Concept for Silicate Planets](https://iopscience.iop.org/article/10.3847/PSJ/adb969). The Radial Profile Analysis System estimates the apparent outer rim diameter of a hypervelocity impact crater using a digital elevation model (DEM) and centroid coordinates as input. Topographic rings surrounding an impact structure are identified through an iterative process and the most consistently identified positive relief ring is selected as the putative outer rim. The Radial Profile Analysis System can be run in the following modes:
- Simple: Predict outer rim using inner cavity diameter estimates.
- Complex: Predict outer rim using outer rim diameter estimates.
- Unknown: Attempt to run in complex mode. Failure to detect an outer rim in complex mode will rerun in simple mode.

## Dependencies
The script requires a DEM projected into a UTM coordinate reference system, Easting and Northing (UTM Coordinates) of the crater center, and an estimate of the inner ring diameter to initialize the search window. The following Python packages are required:
- Pandas
- Geopandas
- Shapely
- Scipy
- Numba
- Numpy
- Rasterio

## Disclaimer
The Radial Profile Analysis System was designed and validated when using specific input parameters. These are provided as user-defined command line inputs, but results are not guaranteed unless the following are used:
- '-pc'/'--peakCount': Number of detected peaks to be considered a potential outer rim. Designed using a value of 4.
- '-a'/'--radiiStep': Step size in degrees to create radii. Designed using a value of 1.
- '-sd'/'--searchDistance': Search distance variant: 1 or 2. Designed using option 1.
- '-pthresh'/'--pthresh': Flag to select all 4th order polynomial elevation topographic rings. Designed to only select 10.

The hardcoded thresholds and search window sizes used within the script were determined through trial and error using a set of reference impact structures and available DEMs. These values may not be extensible to all planetary bodies, impact structure forms, or DEM resolutions.  

The Radial Profile Analysis System searches for circular topographic features within a DEM. An accurate centroid coordinate is critical for the successful determination of the outer rim. The threshold of acceptable positional error in the centroid coordinate has not been determined. It is possible that the Radial Profile Analysis System will fail to identify an inner or outer rim if the centroid coordinate is misplaced. 

A loose estimate of the inner cavity diameter is required to initialize where the Radial Profile Analysis System searches for topographic rings. Significant over or underestimations of this value could result in the Radial Profile Analysis System searching the incorrect location. It is better to overestimate rather than underestimate this parameter. 

## Paper Inputs
Crater centroids used for the selected terrestrial impact structures:
- Barringer: 497928 3876083
- El'gygytgyn: 544170 7486427
- Goat Paddock: 254184 7971237
- Haughton: 425022 8368829
- Logancha: 637001 7268087
- Mistastin: 481363 6193702
- Tswaing: 608886 7189307

# Citation
If you use this code and/or derived products in a scientific publication, please cite:
- Anderson, C. J., Garvin, J. B., McClain, D. R., Sinno, S. S., Melocik, K. A., & Tucker, C. J. (2025). Identifying the Dimensions of Degraded Impact Structures Using Digital Elevation Models: A Proof of Concept for Silicate Planets. The Planetary Science Journal, 6(3), 76.
- Anderson, C., Garvin, J., McClain, D., Sinno, S., Melocik, K., & Tucker, C. (2026). Identifying the Dimensions of Degraded Impact Structures Using Digital Elevation Models: A Proof of Concept for Silicate Planets - Paper Version. In The Planetary Science Journal (Vol. 6, Issue 3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23503865




