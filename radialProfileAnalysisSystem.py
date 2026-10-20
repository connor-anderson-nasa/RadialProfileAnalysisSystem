'''
Author: Connor Anderson, connor.anderson@nasa.gov
Developed for "Identifying the Dimensions of Degraded Impact Structures Using Digital Elevation Models: A Proof of Concept for Silicate Planets"
https://iopscience.iop.org/article/10.3847/PSJ/adb969
'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import rasterio
import math
from scipy import signal
from numba import jit
import numpy as np
import time as runTime
import random
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

## Get new point based on coordinates, distance, and bearing
def point_pos(x0, y0, d, bearing):
    """Get new northing/easting given distance and bearing
    :param x0: easting of initial point
    :param y0: northing of initial point
    :param d: distance in meters
    :param bearing: angle in degrees
    :return x1,y1: easting and northing of new point"""
    
    theta_rad = math.pi/2 - math.radians(bearing)
    return x0 + d*math.cos(theta_rad), y0 + d*math.sin(theta_rad)

## Create sampling points
def create_radial_profile(center_x, center_y, distance, sample_distance, angle_step = 1):
    """Creates points to be sampled for each radial profile
    :param center_x: Easting for central point
    :param center_y: Northing for central point
    :param distance: length in meters for each radial profile
    :param sample_distance: distance at which points are created in meters
    :param angle_step: step in degrees to create transects  -- default is 1
    :retrun lines: geopandas geodataframe containing all the points to be sampled
    """
    lines = []
    for i in np.arange(0,360, angle_step):
        # Create line
        x, y = point_pos(center_x, center_y, distance, i)
        new_line = LineString([(center_x, center_y), (x,y)])
        
        # Create points along line
        start_dist = 0
        add_distance = sample_distance    
        list_of_points = []
        d_steps = []
        while start_dist < new_line.length:
            d_steps.append(start_dist)
            new_point = new_line.interpolate(start_dist)
            list_of_points.append(new_point)
            start_dist += add_distance    
        gdf = gpd.GeoDataFrame(pd.DataFrame({'Angle': [i]* len(list_of_points), 'Distance': d_steps}), geometry = list_of_points).reset_index(drop = True)
        
        if i < 0:
            lines.append(gdf[1:].iloc[::-1])
        else:
            lines.append(gdf)
    return pd.concat(lines).reset_index(drop = True)

## Sample radial profile points
def sample_radial_profile(lines, img, mars = False, venus = False):
    """Samples a raster at points
    :param lines: geopandas geodataframe containing the points to sample
    :param img: dem opened with rasterio
    :param mars: indicates crater on mars
    :param venus: indicates use of rasterio for sampling
    :return lines: geopandas geodataframe of sampled points
    """
    ## Assumes that both vector and raster are in same crs!
    lines = lines.set_crs(img.crs)

    coord_list = [(x,y) for x,y in zip(lines['geometry'].x , lines['geometry'].y)]

    # Numpy sampling is faster, so we try this first
    geotransform = img.transform
    logging.info('Geotransform: \n {}'.format(geotransform))
    rb = img.read(1)
    # Convert each easting/northin to pixel x/y and sample
    try:
        if venus is False:
            pixel_value = []
            n = 0
            for i in coord_list:
                px = math.floor((i[0] - geotransform[2])/geotransform[0])
                py = math.floor((i[1] - geotransform[5])/geotransform[4])
                # py, px = img.index(i[0], i[1])
                if n == 0:
                    print(px,py)
                if (px < rb.shape[0]) & (py < rb.shape[1]):
                    if n == 0:
                        print(rb[py, px])
                    pixel_value.append(rb[py, px])
                else:
                    pixel_value.append(np.nan)
                n+=1
            lines['Elevation'] = [np.max(i) for i in pixel_value]

        elif venus is True:
            logging.info('Venus. Sampling with rasterio')
            pixel_value = []
            for i in img.sample(coord_list):
                pixel_value.append(i)
            lines['Elevation'] = np.concatenate(pixel_value)
            return lines.dropna() 

        if mars is True:
            return lines.dropna() 

        elif np.nanmean(lines[lines.Distance == 0].Elevation.values) > -9000:
                return lines.dropna() 
                
        else:
            logging.warning('Unable to sample raster using numpy. Switching to slower sampling with rasterio')
            pixel_value = []
            for i in img.sample(coord_list):
                pixel_value.append(i)
            lines['Elevation'] = np.concatenate(pixel_value)
            # print(pixel_value)
            return lines.dropna() 

    except IndexError:    
        logging.warning('Index Error: Unable to sample raster using numpy. Switching to slower sampling with rasterio')
        pixel_value = []
        for i in img.sample(coord_list):
            pixel_value.append(i)
        lines['Elevation'] = np.concatenate(pixel_value)
        # print(pixel_value)
        return lines.dropna() 

## Detrend elevations in a radial profile
def detrend_transect(df):
    '''Detrend an elevation profile using scypy signal.detrend
    :param df: geopandas geodataframe containing Elevation column
    :return : linearly detrended elevation values
    '''
    return signal.detrend(df.Elevation, type = 'linear')

## Linear detrend with np.nan
def nan_least_squares(arr, axis = -1, bp=0, overwrite_data=False, type = 'linear'):
    '''scipy signal.detrend that allows for np.nan within the array by replacing them with array average
    :param arr: pandas dataframe where no data elevations have been 
    :param axis: axis which to detrend, default is -1
    :param bp: array-like of ints, Sequence of break points. Optional
    :param overwrite_data: If True, perform in place detrending and avoid a copy. Default is False:
    :return ret: detrended array containing np.nans  
    '''
    arr = arr.Elevation.values
    # print(arr)

    data = np.where(np.isnan(arr), np.nanmean(arr), arr)
    dshape = data.shape
    dtype = data.dtype.char
    N = dshape[-1]
    bp = np.sort(np.unique(np.r_[0, 0, N]))
    Nreg = len(bp) - 1

    # Restructure data so that axis is along first dimension and
    #  all other dimensions are collapsed into second dimension
    rnk = len(dshape)
    if axis < 0:
        axis = axis + rnk

    newdims = np.r_[axis, 0:axis, axis + 1:rnk]
    newdata = np.reshape(np.transpose(data, tuple(newdims)),
                         (N, np.prod(dshape) // N))
    # reshape array with nans
    newdata2 = np.reshape(np.transpose(arr, tuple(newdims)),
                         (N, np.prod(dshape) // N))

    if not overwrite_data:
        newdata = newdata.copy()  # make sure we have a copy
    if newdata.dtype.char not in 'dfDF':
        newdata = newdata.astype(dtype)
    # Find leastsq fit and remove it for each piece
    for m in range(Nreg):
        Npts = bp[m + 1] - bp[m]
        A = np.ones((Npts, 2), dtype)
        # A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
        A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
        sl = slice(bp[m], bp[m + 1])
        if type == 'linear':
            coef, resids, rank, s = np.linalg.lstsq(A, newdata[sl], rcond = -1)
        elif type == 'poly':
            coef, resids, _ = np.polyfit(A, newdata[sl], 4)
        newdata[sl] = newdata2[sl] - np.dot(A, coef)

    # Put data back in original shape.
    tdshape = np.take(dshape, newdims, 0)
    ret = np.reshape(newdata, tuple(tdshape))
    vals = list(range(1, rnk))
    olddims = vals[:axis] + [0] + vals[axis:]
    ret = np.transpose(ret, tuple(olddims))

    return ret

## Adjust elevations relative to initial point (distance == 0)
def set_initial_point(df):
    '''Adjust elevations relative to point at center of crater
    :param df: geopandas geodataframe containing elevation profiles
    :return numpy array: detrended elevations relative to crater center
    '''
    intitial_value = df[df.Distance == 0].Detrend.values[0]
    return df.Detrend - intitial_value

## Create a polynomial fit without nan values
def nan_poly_fit(df, order):
    '''Fit a polynomial function using distance from crater center and elevation - drops no data before running
    :param df: geopandas geodataframe containing elevation profiles
    :return : nth order polynomial coefficients
    '''  
    df = df.dropna()
    coeff = np.polyfit(df.Distance.values, df.Elevation.values, order)
    return coeff

# Fit and apply polynomial function
def poly_apply(df, order):
    '''fit and apply a polynomial equation
    :param df: geopandas geodataframe containing elevation profiles
    :return : predicted elevation values using nth order polynomial
    '''
    coeff = nan_poly_fit(df, order)
    return np.polyval(coeff, df.Distance.values)

def set_initial_poly_point(df):
    '''Adjust polynomial elevations relative to point at center of crater
    :param df: geopandas geodataframe containing elevation profiles
    :return numpy array: detrended elevations relative to crater center
    '''
    intitial_value = df[df.Distance == 0].poly_detrend.values[0]
    return df.poly_detrend - intitial_value

## Checks if lists are same length, Not used here, but keeping for transparency 
def _datacheck_peakdetect(x_axis, y_axis):
    '''Used in peakdetect(), and pulled from same source.Checks to make sure that x_axis and y_axis are arrays of the same length.
    :param x_axis: numpy array
    :param y_axis: numpy array
    :return x_axis, y_axis: numpy arrays containing x and y axis values for peakdetect
    '''
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

@jit(nopython= True)
## Function pulled to detect local maxima/minma in data
def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Pulled from: https://gist.github.com/sixtenbe/1178136

    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data is same length 
    # NOTE Connor: Our arrays should be the same length so we skip this step.
    # x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)

    # store data length for later use
    length = len(y_axis)

    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.inf, -np.inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.inf
                mn = np.inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
    
        ###look for min####
        if y > mn+delta and mn != -np.inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.inf
                mx = -np.inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
    
    #Remove the false hit on the first value of the y_axis
    if len(dump) > 0 and dump[0]:
        max_peaks.pop(0)
    else:
        if len(min_peaks) > 0:
            min_peaks.pop(0)
    return np.asarray([n[0] for n in max_peaks])

@jit(nopython = True)
## Get count of values within a range, returns the sum of True values within a boolean array.
def count_values_within_range(df, min_val, max_val):
    """Counts number of rows within a search window
    :param df: array of distance values - pd/gpd dataframe containing Distance column
    :param min_val: Minimum value of search window, int
    :param max_val: Maximum value of search window, int
    :return count: number of rows within search window 
    """
    return ((min_val < df) & (df < max_val)).sum()

## Counts number of peaks within each 100 m window
@jit(nopython = True)
def peaks_by_distance(distances_array, distance, window_size, angle):
    """Counts the number of peaks within each 100 m window
    :param distance_array: numpy array distances for local maxima
    :param distance: search radius in meters
    :param window_size: matching window size in meters
    :param angle: transect identifying angle
    :return intervals: df containing the count of peaks per specified window size
    """
    intervals_start = np.arange(0, distance, window_size)
    intervals_end = np.arange(window_size, distance + window_size, window_size)
    intervals_counts = np.asarray([count_values_within_range(distances_array, x, y) for x,y in zip(intervals_start, intervals_end)])
    intervals_counts = np.where(intervals_counts > 0, 1, 0) 
    intervals_angle = np.asarray(len(intervals_counts) * [angle]) 

    return intervals_start, intervals_end, intervals_counts, intervals_angle

## Round a number to the nearest 100
def roundup(x):
    ''' Round number to nearest 100
    :param x: number, int or float
    :return val: rounded number
    '''
    return x if x % 100 == 0 else x + 100 - x % 100

## Create array containing detrended elevations and matching indexes
def set_up_peak_search(df, detrend_type):
    '''Creates a list of tuples containing arrays for the transect angle, detrended elevation, and index value
    :param df: pandas dataframe containing detrended elevations.
    :detrend_type: str, type of detrened elevation, 'Adj_Detrend' or 'Polynomial'
    :return : list of tuples
    '''
    angle_intervals = df.Angle.unique()
    angles = []
    yAxisArrs = []
    xAxisArrs = []
    if detrend_type == 'Adj_Detrend':
        for i in angle_intervals:
            angles.append(i)
            yAxisArrs.append(df[df.Angle == i].Adj_Detrend.values)
            xAxisArrs.append(np.asarray(range(len(df[df.Angle == i].Adj_Detrend.values))))    
    if detrend_type == 'Polynomial':
        for i in angle_intervals:
            angles.append(i)
            yAxisArrs.append(df[df.Angle == i].poly_detrend.values)
            xAxisArrs.append(np.asarray(range(len(df[df.Angle == i].Adj_Detrend.values))))   

    return angles, yAxisArrs, xAxisArrs

## Search for peaks
@jit(nopython = True)
def search_for_peaks(angles, yAxisArrs, xAxisArrs, r0, r1):
    '''Wrapper function to search all detrended elevation profiles for peaks
    :param interval_data: deprecated- list of tuples containing the detrended elevations and indexes for each transect angle
    :param angles: numpy array containing the angle of each radial profile
    :param yAxisArrs: numpy array containing the elevations of each radial profile
    :param xAxisArrs: numpy array containing the indicies of each radial profile
    :param r0: lookahead value
    :param r1: delta value
    :return transect angle: list of angle associated with each peak
    :return detected_peaks: list of indicies for each detected peak
    '''
    detected_peaks = []
    transect_angle = []

    for i in range(len(angles)):
        maxp = peakdetect(y_axis = yAxisArrs[i], x_axis = xAxisArrs[i], lookahead = r0, delta= r1)
        if len(maxp) > 0:
            detected_peaks.append(maxp)
            transect_angle.append(angles[i])          
    
    return transect_angle, detected_peaks ## Returns the angle and index for the peak in two lists of arrays

## Perform iterative search for local maxima
def iterative_peak_search(lines, max_lookahead, search_distance, window_size, p_count = 4, pthresh = False):
    """Iterates through a number of lookahead values for peakdetect()
    First search is completed using the adjusted detrend values which places the crater center
    at 0 m elevation. Second search is completed using the fourth order polynomial detrend. Only
    peak distances which have more than four occurances across all radii of an iteration are reported.
    Peaks identified with the adjusted detrend values are weighted higher than those identified using
    the polynomial detrend (2-1). 

    :param lines: pd/gpd dataframe containing all of the generated points with sampled elevations
    :param max_lookahead: maximum lookahead distance based on raster cell size. Maximum distance is 500 m
    :param search_distance: maximum distance in meters to count peaks, used in peaks_by_distance()
    :param window_size: matching window size in meters -- used in peaks_by_distance()
    :param pthresh: args.pthresh, indicates how many rings to use with polynomial detection
    :return potential_ring_distances: list containing all identified peak distances across all iterations, not currently used once reported
    :return pd df: pandas dataframe containing all identified peaks detected in at least five transects during an iteration
    """
    potential_ring_distances = []
    peak_df = []
    param_array = np.array(np.meshgrid(np.arange(1, max_lookahead + 1,1), 0.1)).T.reshape(-1, 2)   
    logging.info('Starting window: {}'.format(window_size))
    angles, yAxisArrs, xAxisArrs = set_up_peak_search(lines, detrend_type = 'Adj_Detrend')

    for r in param_array:
        transect_angles, max_peaks = search_for_peaks(angles, yAxisArrs, xAxisArrs, r0 = int(r[0]), r1 = int(r[1]))
        max_peaks = [(x, y) for x,y in zip(transect_angles, max_peaks)]
        potential_peaks = []

        for p in max_peaks:
            potential_peaks.append(lines[lines.Angle == p[0]].iloc[p[1]])
        potential_peaks = pd.concat(potential_peaks)

        all_start, all_end, all_counts, all_angles = zip(*[peaks_by_distance(potential_peaks[potential_peaks.Angle == ang].Distance.values,
                                                                            search_distance,
                                                                            window_size,
                                                                            angle = ang) for ang in potential_peaks.Angle.unique()])
        t = pd.DataFrame({'Start': np.concatenate(all_start), 
                          'End': np.concatenate(all_end), 
                          'Count': np.concatenate(all_counts), 
                          'Angle': np.concatenate(all_angles)})

        t = t.groupby(by = 'End')['Count'].sum().reset_index(name='Count').sort_values(by = 'Count', ascending = False).reset_index(drop=True)
        # Only select peaks that appear in more than p_count transects
        t = t[t.Count > p_count].reset_index(drop = True) 
        potential_ring_distances.append(np.repeat(t[t.Count > p_count].End.values, 2))

        ## Set up to export all peak counts
        t['Lookahead'] = len(t.Count.values) * [r[0]]
        t['Type'] = len(t.Count.values) * ['CAD']
        peak_df.append(t)

    # Running peak detection with polynomial detrended values
    angles, yAxisArrs, xAxisArrs = set_up_peak_search(lines, detrend_type = 'Polynomial')
    for r in param_array:
        transect_angles, max_peaks = search_for_peaks(angles, yAxisArrs, xAxisArrs, r0 = int(r[0]), r1 = int(r[1]))

        max_peaks = [(x, y) for x,y in zip(transect_angles, max_peaks)]
        potential_peaks = []
        for p in max_peaks:
            potential_peaks.append(lines[lines.Angle == p[0]].iloc[p[1]])
        potential_peaks = pd.concat(potential_peaks)
    
        all_start, all_end, all_counts, all_angles = zip(*[peaks_by_distance(potential_peaks[potential_peaks.Angle == ang].Distance.values,
                                                                             search_distance, 
                                                                             window_size, 
                                                                             angle = ang) for ang in potential_peaks.Angle.unique()])
        t = pd.DataFrame({'Start': np.concatenate(all_start), 
                          'End': np.concatenate(all_end), 
                          'Count': np.concatenate(all_counts), 
                          'Angle': np.concatenate(all_angles)})        


        # NOTE Connor 11/9/2024: The RPS performs better when only selecting the 10 most frequently ID'd topographic rings from the polynomial detrend values
        # NOTE Connor 11/12/2024: Added arg to switch between number of topographic rings to select. Default is to not use args.pthresh
        if pthresh is True:
            t = t.groupby(by = 'End')['Count'].sum().reset_index(name='Count').sort_values(by = 'Count', ascending = False).reset_index(drop=True)
        else:
            t = t.groupby(by = 'End')['Count'].sum().reset_index(name='Count').sort_values(by = 'Count', ascending = False).reset_index(drop=True)[:10]
        t = t[t.Count > p_count].reset_index(drop = True)
        potential_ring_distances.append(t[t.Count > p_count].End.values)
        
        # Set up to export all peak counts
        t['Lookahead'] = len(t.Count.values) * [r[0]]
        t['Type'] = len(t.Count.values) * ['PND']
        peak_df.append(t)   

    peak_df = pd.concat(peak_df).reset_index(drop=True)
    peak_df['Window'] = len(peak_df) * [window_size]     

    return peak_df

## Find lowest count
def find_difference(df, val_a, val_b):
    """Given two distances, reports the one with the lowest frequency
    :param df: pd dataframe containing distances and frequencies
    :param a,b: distances
    :return distance: distance with lowest frequency
    """
    if (abs(val_a-val_b)) < 1000: # sanity check to make sure one value isn't no data
        mfp = min(df[df.Distance == val_a].Adj_Count.values[0], df[df.Distance == val_b].Adj_Count.values[0])
        return df[df.Adj_Count == mfp].Distance.values[0]

## Get count of values within a range
@jit(nopython = True)
def count_counts(df, min_val, max_val):
    """Counts number of rows within a search window
    :param df: numpy array containing the pd/gpd dataframe Distance column values
    :param min_val: Minimum value of search window, int
    :param max_val: Maximum value of search window, int
    :return count: number of rows within search window 
    """
    return np.sum((min_val <= df) & (df < max_val)) 

## Counts number of peaks within each window
def peaks_by_count(df, distance, window_size):
    """Counts the number of peaks within each window
    :param df: df containing distances for local maxima
    :param distance: search radius in meters
    :param window_size: matching window size in meters
    :return intervals: df containing the count of peaks per specified window size
    """
    try:
        intervals = pd.DataFrame({'Start': np.arange(0, distance, window_size), 'End': np.arange(window_size, distance + 1, window_size)})
        df_distance = df.Distance.values
        intervals['Adj_Count'] = [count_counts(df_distance, x, y) for x,y in zip(intervals.Start, intervals.End)]  
        return intervals
    except ValueError:
        intervals = pd.DataFrame({'Start': np.arange(0, distance, window_size), 'End': np.arange(window_size, distance + window_size, window_size)})
        df_distance = df.Distance.values
        intervals['Adj_Count'] = [count_counts(df_distance, x, y) for x,y in zip(intervals.Start, intervals.End)]
        return intervals

## Estimates crater diameter using iteration results
def finalRingSorting(df, inner_cavity_estimate, p_count, count_threshold, mars = False, vSmall = False):
    """Attempts to sort through all iteration results to identify the most probably crater diameters
    :param df: pandas dataframe containing the concatenated results from all peak detection iterations
    :param inner_cavity_estimate: Estimate of inner cavity radius in meters (args.diameter/2)
    :param count_threshold: Value used to select potential outer rims. Count of detections must be within this value of the
                            probable outer rim to be considered part of the outer rim range.
    :param mars: Indicates crater is on Mars
    :param vSmall: Indicates crater is believed to be less than 250 m in diameter
    :returns possible_rings: Range of potential outer rims with their detected frequency
    """
    df = df[df.Count > p_count].reset_index(drop = True)
    df['Distance'] = df.End.values
    df = df[['Distance', 'Type', 'Count']]
    # Double count of CAD-detected peaks to weight it 2-1
    df = pd.concat([df, df[df.Type == 'CAD']])

    if mars is True:
        possible_rings = peaks_by_count(df, df.Distance.max(), 2000)
    elif vSmall is True:
        possible_rings = peaks_by_count(df, df.Distance.max(), 20)
    else:
        possible_rings = peaks_by_count(df, df.Distance.max(), 200)
    possible_rings = possible_rings.groupby(by = 'End')['Adj_Count'].sum().reset_index(name='Count').reset_index(drop=True)
    possible_rings = pd.DataFrame({'Distance': possible_rings.End.values, 'Adj_Count': possible_rings.Count.values})
    ic_diff = list(1 - abs(possible_rings[possible_rings.Distance < inner_cavity_estimate].Distance - inner_cavity_estimate)/inner_cavity_estimate)
    a = [(x,y) for x,y in zip(possible_rings[possible_rings.Distance < inner_cavity_estimate].Adj_Count, ic_diff)]
    possible_rings['Structure'] = len(possible_rings) * ['nan']
    logging.info('{}/{} matching windows had no peaks.'.format(len(possible_rings[possible_rings.Adj_Count == 0]), len(possible_rings.index)))
    possible_rings = possible_rings[possible_rings.Adj_Count != 0].reset_index(drop=True)

    if mars is True:
        upper_limit = inner_cavity_estimate + 2500
        lower_limit = inner_cavity_estimate - 5000
    elif vSmall is True:
        upper_limit = inner_cavity_estimate + 25
        lower_limit = inner_cavity_estimate - 50
    else:
        upper_limit = inner_cavity_estimate + 250
        lower_limit = inner_cavity_estimate - 500


    possible_rings['Structure'] = np.where(possible_rings.Distance == possible_rings[(possible_rings.Distance <= upper_limit) &
                                        (possible_rings.Distance > lower_limit) &
                                        (possible_rings.Adj_Count == possible_rings[(possible_rings.Distance <= upper_limit) &
                                        (possible_rings.Distance > lower_limit)].Adj_Count.max())].Distance.values[0],
                                        'Inner Cavity', 'nan')

    ic_radius = possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]
    ic_count = possible_rings[possible_rings.Structure == 'Inner Cavity'].Adj_Count.values[0]
    logging.info('Inner Cavity Radius: {}'.format(ic_radius))
    logging.info('Inner Cavity Radius Count: {}'.format(ic_count))

    # Find all IC diameters within range
    if mars is True:
        possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                               (possible_rings.Distance <= ic_radius + 6000) &
                                               (possible_rings.Distance >= ic_radius - 6000) & 
                                               (possible_rings.Adj_Count > (ic_count-count_threshold)), 
                                               'IC Range', possible_rings.Structure)

    elif vSmall is True:
        possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                               (possible_rings.Distance <= ic_radius + 30) &
                                               (possible_rings.Distance >= ic_radius - 30) & 
                                               (possible_rings.Adj_Count > (ic_count-count_threshold)), 
                                               'IC Range', possible_rings.Structure)

    else:
        possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                               (possible_rings.Distance <= ic_radius + 600) &
                                               (possible_rings.Distance >= ic_radius - 600) & 
                                               (possible_rings.Adj_Count > (ic_count-count_threshold)), 
                                               'IC Range', possible_rings.Structure)
    # Find the smallest diameter to use in calculating OR search window
    ic_min = possible_rings[possible_rings.Structure != 'nan'].Distance.min()
    ic_max = possible_rings[possible_rings.Structure != 'nan'].Distance.max()
    logging.info('Using Inner Cavity Range for Outer Rim Range: {} to {}'.format(ic_min, ic_max))

    # Calculate lower threshold
    if mars is True:
        lower_OR_threshold = 1.5*ic_min + 15000
        upper_OR_threshold = 2.2*ic_max + 15000
    elif vSmall is True:
        lower_OR_threshold = 1.5*ic_min + 15
        upper_OR_threshold = 2.2*ic_max + 15        
    else:
        lower_OR_threshold = 1.5*ic_min + 1500
        upper_OR_threshold = 2.2*ic_max + 1500


    logging.info('Calculated Outer Rim Range: {} to {}'.format(lower_OR_threshold, upper_OR_threshold))

    # Identify the most frequent peak within OR threshold
    OR_count = possible_rings[(possible_rings.Distance < upper_OR_threshold) & (possible_rings.Distance >= lower_OR_threshold)].Adj_Count.max()

    possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                           (possible_rings.Distance < upper_OR_threshold) & 
                                           (possible_rings.Distance >= lower_OR_threshold) & 
                                           (possible_rings.Adj_Count == OR_count), 
                                           'Outer Rim', possible_rings.Structure)

    # Select peaks to build estimation range
    possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                           (possible_rings.Distance < upper_OR_threshold) & 
                                           (possible_rings.Distance >= lower_OR_threshold) & 
                                           (possible_rings.Adj_Count > OR_count-count_threshold), 
                                           'OR Range', possible_rings.Structure)

    logging.info('Estimated Rims and Potential Ranges:')
    print(possible_rings[(possible_rings.Structure == 'Outer Rim') | (possible_rings.Structure == 'OR Range')])

    return possible_rings

## Get the range of values within list
def get_range(val_list):
    '''
    :param val_list: list of numbers
    :return : maximum value minus minimum value
    '''
    return np.nanmax(val_list) - np.nanmin(val_list)

## Get count of NaN within list
def count_nan(val_list):
    '''Get count of no data values
    :param val_list: list of values
    :return : count of no data values
    '''
    return val_list.count(np.nan)

def main():
    parser = argparse.ArgumentParser(description='Radial Profile Analysis System', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--inputDEM', type=os.path.abspath, metavar='PATH', required = True, 
                        help="""Path to DEM to be used for calculating ring diameters.""")
    parser.add_argument('-c', '--centroid', nargs='+', type=int, required = True, metavar='EASTING NORTHING', 
                        help="""Easting and Northing for crater center, e.g. '237740 2720414'.""")
    parser.add_argument('-t', '--craterType',type=str, required = True, metavar='craterType',
                        help="""Crater type, i.e., simple, complex, or unknown.
                        Simple: Predict outer rim using inner cavity diameter estimates.
                        Complex: Predict outer rim using outer rim diameter estimates.
                        Unknown: Attempt to run in complex mode. Failure to detect an outer rim in complex mode will rerun in simple mode.""")
    parser.add_argument('-d', '--diameter', type=int, required = True, metavar='DIAMETER',
                        help="""\
    Loose estimate for inner cavity diameter in meters.
    It is better to overestimate than underestimate this parameter.""")
    parser.add_argument('-w', '--windowSize', type=int, metavar='WINDOW',
                        help="""Matching window size in meters.
                        Used to count the number of peaks.""")
    parser.add_argument('-pc', '--peakCount', type=int, metavar='PEAKCOUNT',
                        help="""Number of detected peaks to be considered a potential outer rim.
                        Used to filter out individual peaks.""")
    parser.add_argument('-a', '--radiiStep', type=float, required = True, metavar='DEGREES',
                        help="""Step size in degrees to create radii. If unsure, set to 1.""")
    parser.add_argument('-o', '--outPath', type=os.path.abspath, required = True, metavar='PATH',
                        help="""Path to output file with .shp extension.""")
    parser.add_argument('-sd', '--searchDistance', required = True, type=int, metavar='INT',
                        help="""\
    Search distance variant: 1 or 2.
    1: Iterate through each of the following search distance variants- 
    - roundup((args.diameter/2) * 2.2 + 2000)
    - roundup((args.diameter/2) * 3)
    - roundup((args.diameter/2) * 4)
    - roundup((args.diameter/2) * 5)
    2: Iterate through n random number of search distances. Currently set to 100.""")
    parser.add_argument('-mars', '--mars', action='store_true',
                        help="""Flag to adjust matching window sizes to account for scaling difference on Mars.""") 
    parser.add_argument('-venus', '--venus', action='store_true',
                        help="""Flag to sample using rasterio and not numpy.""") 
    parser.add_argument('-pthresh', '--pthresh', action='store_true',
                        help="""Flag to select all 4th order polynomial elevation topographic rings. Default is to select 10.""") 
    parser.add_argument('-parallel', '--parallel', action='store_true',
                        help="""Flag to indicate parallel processing with Python's Parallel module. Default is no parallel processing.""")

    args = parser.parse_args()
    timeStart = runTime.time()
    img = rasterio.open(args.inputDEM)
    geotransform = img.transform
    center_x, center_y = args.centroid[0], args.centroid[1]
    # calculate parameters for future use
    # search distance
    if args.searchDistance == 1:
        search_distance_2 = roundup((args.diameter/2) * 2.2 + 2000)
        search_distance_3 = roundup((args.diameter/2) * 3)
        search_distance_4 = roundup((args.diameter/2) * 4)
        search_distance_5 = roundup((args.diameter/2) * 5)
        search_distances = [search_distance_5, search_distance_4, search_distance_3, search_distance_2]
        search_distances = sorted(search_distances, reverse = True)
        print(search_distances)

    elif args.searchDistance == 2:
        ll = int(roundup((args.diameter/2) * 2.2 + 2000))
        ul = int(roundup((args.diameter/2) * 5))
        num_samples = 100
        search_distances = sorted(random.sample(range(ll, ul), num_samples), reverse = True)

        while len(set(search_distances)) != num_samples:
            num_off = num_samples - len(set(search_distances))
            new_nums = random.sample(range(ll, ul), num_off)
            for i in range(len(new_nums)):
                search_distances.append(new_nums[i])
        search_distances = sorted(search_distances, reverse = True)

        logging.info('Random search distances: {}'.format(search_distances))
              
    # maximum lookahead of 500 m based on cell size
    if args.diameter < 500:
        max_lookahead = 25
    elif geotransform[0] < 200:
        max_lookahead = 500/geotransform[0]
    elif geotransform[0] >= 200:
        max_lookahead = 25
    
    # Set matching window sizes
    vSmall = False
    if args.diameter < 250:
        vSmall = True
        matching_windows = [10, 15, 20, 25, 30, 35]
    elif args.mars == True:
        matching_windows = [1000, 1500, 2000, 2500, 3000, 3500]
    else:
        matching_windows = [100, 150, 200, 250, 300, 350]


    # Calculate value to use for frequency
    freq_denominator = max_lookahead * 3
    logging.info('Parameters for radial profile analysis:')
    logging.info('Running in {} mode. Expected crater type is {}.'.format(args.craterType, args.craterType))
    logging.info('Center Easting: {}'.format(center_x))
    logging.info('Center Northing: {}'.format(center_y))
    logging.info('Search Distance Variant: {}'.format(args.searchDistance))
    logging.info('DEM Resolution: {}'.format(geotransform[0]))
    logging.info('Maximum Lookahead: {}'.format(max_lookahead))
    logging.info('Matching Window Size {} m'.format(args.windowSize))
    if (args.peakCount):
        logging.info('Peak Count Threshold: {} matches'.format(args.peakCount))
        p_count = args.peakCount
    else:
        logging.info('No peak count threshold specified. Defaulting to > 4 peaks.')
        p_count = 4
    logging.info('Making radii every {} degrees'.format(args.radiiStep))
    if geotransform[0] >= 50:
        count_threshold = 30
        logging.info('Image Resolution = {}. Using Count Threshold of 30'.format(geotransform[0]))
    elif (geotransform[0] < 50) & (geotransform[0] >= 30):
        count_threshold = 25
        logging.info('Image Resolution = {}. Using Count Threshold of 25'.format(geotransform[0]))    
    elif (geotransform[0] < 30) & (geotransform[0] > 4):
        count_threshold = 65
        logging.info('Image Resolution = {}. Using Count Threshold of 65'.format(geotransform[0]))
    elif geotransform[0] == 4:
        count_threshold = 125
        logging.info('Image Resolution = {}. Using Count Threshold of 125'.format(geotransform[0]))
    elif geotransform[0] < 4:
        count_threshold = 250
        logging.info('Image Resolution = {}. Using Count Threshold of 210'.format(geotransform[0]))
    logging.info('Matching window sizes: {}'.format(matching_windows))
    if args.pthresh:
        logging.info('RPS will select all topogrpahic rings within the 4th order polynomial detrended elevations.')
        logging.info('This may lead to less accurate results.')
        pthresh = True
    else:
        logging.info('Using default selection of ten topographic rings from 4th order polynomial detrended elevations.')
        pthresh = False
    if args.parallel:
        from joblib import Parallel, delayed, cpu_count
        logging.info('Running using the Parallel module in Python.')

    combined_rings = []

    n = 1

    ## Main analysis ##
    if (args.searchDistance == 1) | (args.searchDistance == 2):
        logging.info('Generating sample points using maximum sampling distance of {}.'.format(search_distances[0]))
        lines = create_radial_profile(center_x = center_x,
                                      center_y = center_y,
                                      distance = search_distances[0],
                                      sample_distance = geotransform[0],
                                      angle_step = args.radiiStep)
        logging.info('Sampling generated points.')
        lines = sample_radial_profile(lines,
                                      img,
                                      mars = args.mars,
                                      venus = args.venus)

        lines = lines[lines.Elevation != -9999]
        lines = lines[lines.Elevation != 0]
        
        for x in search_distances:
            logging.info('Starting Search Distance: {}'.format(x))
            lines = lines[lines.Distance <= x]
            logging.info('Calculating metrics to use for local maxima search.')
            lines['Detrend'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: nan_least_squares(lines)).values)
            lines['Adj_Detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: set_initial_point(lines)).values
            lines['poly_fit'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: poly_apply(lines, 4)).values)
            lines['poly_detrend'] = lines.Elevation - lines.poly_fit
            lines.Adj_Detrend = np.where(lines.Distance == 0, np.nan, lines.Adj_Detrend)
            lines.poly_detrend = np.where(lines.Distance == 0, np.nan, lines.poly_detrend)
            
            iteration_rings = []
            # if args.mars == True:
            #     swds = [1000, 1500, 2000, 2500, 3000, 3500]
            # else:
            #     swds = [100, 150, 200, 250, 300, 350]

            if args.parallel:
                iteration_rings = Parallel(n_jobs = 6, verbose = 1)(delayed(iterative_peak_search)(lines = lines, 
                                                                                                   max_lookahead = max_lookahead, 
                                                                                                   search_distance = x, 
                                                                                                   window_size = i, 
                                                                                                   p_count = p_count,
                                                                                                   pthresh = pthresh) for i in matching_windows)
            else:
                # For processing without Python's parallel module
                iteration_rings = [iterative_peak_search(lines = lines,
                                                         max_lookahead = max_lookahead,
                                                         search_distance = x,
                                                         window_size = i,
                                                         p_count = p_count,
                                                         pthresh = pthresh) for i in matching_windows]

            iteration_rings = pd.concat(iteration_rings).reset_index(drop=True)  
            final_diameters_all = finalRingSorting(df = iteration_rings,
                                                        inner_cavity_estimate = args.diameter/2,
                                                        p_count = p_count,
                                                        count_threshold = count_threshold,
                                                        mars = args.mars,
                                                        vSmall = vSmall)  
            final_diameters_all['Structure'] = np.where(final_diameters_all.Structure == 'nan', 'False Positive', final_diameters_all.Structure)  
            final_diameters_all.to_csv('{}_{}.csv'.format(args.outPath[:-4], x), index=False)
            combined_rings.append(final_diameters_all)

        logging.info('Combining results across each search distance.')
        logging.info('Beginning calculation of outer rim prediction interval.')

        # Use topographic rings outside of the inner ring diameter estimate if it is a complex crater
        if (args.craterType == 'complex') | (args.craterType == 'unknown'):            
            outer_rim_results = []
            for results in combined_rings:
                a = results[results.Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
                outer_rim_results.append(a[(a.Structure != 'IC Range') & (a.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True))

            u_dist= []
            for i in outer_rim_results:
                for x in i.Distance.unique():
                    u_dist.append(x)
   
            df = []
            for x in list(set(u_dist)):
                print(x)
                dist_counts = []
                for i in outer_rim_results:
                    try:
                        dist_counts.append(i[i.Distance == x].reset_index(drop=True).Adj_Count[0])
                    except KeyError:
                        dist_counts.append(np.nan)
                df.append(pd.DataFrame({'Diameter': [x*2] , 'Range': [dist_counts]}))

            df = pd.concat(df).reset_index(drop=True)            
            df['Mean_Count'] = [np.nanmean(i) for i in df.Range]
            df['Count_Range'] = [get_range(i) for i in df.Range]
            df['Nan_Count'] = [count_nan(i) for i in df.Range]

            df = df[df.Nan_Count == 0].sort_values(by = 'Count_Range', ascending = True).reset_index(drop=True)
            logging.info('Outer Rim Prediction Interval:')
            print(df)
                
            if (args.craterType == 'complex'):
                df.to_csv('{}_OR_Interval.csv'.format(args.outPath[:-4]), index=False)

            # If outer rim detection failed to find an outer rim, select from inner cavity diameters
            elif (args.craterType == 'unknown'):
                if not df.empty:
                    logging.info('Outer rim found in Unknown mode. Writing predictions.')
                    df.to_csv('{}_OR_Interval.csv'.format(args.outPath[:-4]), index=False)
                elif df.empty:
                    logging.info('Outer rim not found in Unknown mode. Selecting outer rim from possible inner cavity diameters')
                    outer_rim_results = []
                    for results in combined_rings:
                        a = results[results.Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
                        outer_rim_results.append(a[(a.Structure != 'OR Range') & (a.Structure != 'Outer Rim')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True))
                    
                    u_dist= []
                    for i in outer_rim_results:
                        for x in i.Distance.unique():
                            u_dist.append(x)
           
                    df = []
                    for x in list(set(u_dist)):
                        print(x)
                        dist_counts = []
                        for i in outer_rim_results:
                            try:
                                dist_counts.append(i[i.Distance == x].reset_index(drop=True).Adj_Count[0])
                            except KeyError:
                                dist_counts.append(np.nan)
                        df.append(pd.DataFrame({'Diameter': [x*2] , 'Range': [dist_counts]}))

                    df = pd.concat(df).reset_index(drop=True)            
                    df['Mean_Count'] = [np.nanmean(i) for i in df.Range]
                    df['Count_Range'] = [get_range(i) for i in df.Range]
                    df['Nan_Count'] = [count_nan(i) for i in df.Range]

                    df = df[df.Nan_Count == 0].sort_values(by = 'Count_Range', ascending = True).reset_index(drop=True)
                    logging.info('Outer Rim Prediction Interval:')
                    print(df)
                    df.to_csv('{}_OR_Interval.csv'.format(args.outPath[:-4]), index=False)
    
        # Use only the inner cavity diameter estimates if it is a simple crater
        elif (args.craterType == 'simple'):
            outer_rim_results = []
            for results in combined_rings:
                a = results[results.Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
                outer_rim_results.append(a[(a.Structure != 'OR Range') & (a.Structure != 'Outer Rim')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True))
            
            u_dist= []
            for i in outer_rim_results:
                for x in i.Distance.unique():
                    u_dist.append(x)
   
            df = []
            for x in list(set(u_dist)):
                print(x)
                dist_counts = []
                for i in outer_rim_results:
                    try:
                        dist_counts.append(i[i.Distance == x].reset_index(drop=True).Adj_Count[0])
                    except KeyError:
                        dist_counts.append(np.nan)
                df.append(pd.DataFrame({'Diameter': [x*2] , 'Range': [dist_counts]}))

            df = pd.concat(df).reset_index(drop=True)            
            df['Mean_Count'] = [np.nanmean(i) for i in df.Range]
            df['Count_Range'] = [get_range(i) for i in df.Range]
            df['Nan_Count'] = [count_nan(i) for i in df.Range]

            df = df[df.Nan_Count == 0].sort_values(by = 'Count_Range', ascending = True).reset_index(drop=True)
            logging.info('Outer Rim Prediction Interval:')
            print(df)
            df.to_csv('{}_OR_Interval.csv'.format(args.outPath[:-4]), index=False)    

    # Write out execution commands
    execArgs = []
    execArgs.append('InputDEM: {}'.format(args.inputDEM))
    execArgs.append('OutPath: {}_OR_Interval.csv'.format(args.outPath[:-4]))
    execArgs.append('Centroid: {}'.format(args.centroid))
    execArgs.append('CraterType: {}'.format(args.craterType))
    execArgs.append('Diameter: {}'.format(args.diameter))
    execArgs.append('WindowSize: {}'.format(matching_windows))
    execArgs.append('PeakCount: {}'.format(args.peakCount))
    execArgs.append('StepSize: {}'.format(args.radiiStep))
    execArgs.append('SearchDistanceVariant: {}'.format(args.searchDistance))
    if args.pthresh:
        execArgs.append('Pthresh: True')
    else: 
        execArgs.append('Pthresh: False')
    if args.mars:
        execArgs.append('Mars: True')
    else:
        execArgs.append('Mars: False')
    if args.venus:
        execArgs.append('Venus: True')
    else:
        execArgs.append('Venus: False')

    with open('{}_execArgs.txt'.format(args.outPath[:-4]), 'w') as f:
        for line in execArgs:
            f.write(f"{line}\n") 

    timeElapsedTotal = runTime.time() - timeStart
    timeElapsedMin = np.floor(timeElapsedTotal / 60)
    timeElapsedSec = timeElapsedTotal % 60
    logging.info('Script Completed in %d min %d sec.' % (timeElapsedMin, timeElapsedSec))

if __name__ == "__main__":
    main()