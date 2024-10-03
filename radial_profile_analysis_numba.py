from scipy import signal
import pandas as pd
import os
from numba import jit, prange
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
import math
import logging
import argparse
from operator import itemgetter
import time as runTime
import random

from joblib import Parallel, delayed, cpu_count


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
def create_radial_profile(center_x, center_y, distance, sample_distance, angle_step):
    """Creates points to be sampled for each radial profile
    :param center_x: Easting for central point
    :param center_y: Northing for central point
    :param distance: length in meters for each radial profile
    :param sample_distance: distance at which points are created in meters
    :param angle_step: step in degrees to create transects  -- originally set to 1
    :retrun lines: geopandas geodataframe containing all the points to be sampled
    """
    lines = []
    for i in np.arange(0,360, angle_step):
        # Create line
        # x, y = PointFromDistanceBearing(center_x, center_y, 50, i)
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
def sample_radial_profile(lines, img, venus = False):
    """Samples a raster at points
    :param lines: geopandas geodataframe containing the points to sample
    :param img: dem opened with rasterio
    :return lines: geopandas geodataframe of sampled points
    """
    coord_list = [(x,y) for x,y in zip(lines['geometry'].x , lines['geometry'].y)]

    # try sampling with numpy first
    # Get raster information
    geotransform = img.transform
    print(geotransform)
    rb = img.read(1)
    # # Convert each easting/northin to pixel x/y and sample
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
        # print(pixel_value)
        return lines.dropna() 


    if np.nanmean(lines[lines.Distance == 0].Elevation.values) > -9000:
        return lines.dropna() 

    else:
        logging.warning('Unable to sample raster using numpy. Switching to slower sampling with rasterio')
        pixel_value = []
        for i in img.sample(coord_list):
            pixel_value.append(i)
        lines['Elevation'] = np.concatenate(pixel_value)
        # print(pixel_value)
        return lines.dropna() 

## calculate slope using neighboring points
def calc_slope(df):

    return abs((df.Adj_Detrend.diff()/20)*100)

## Normalize the adjusted detrended elevation
def calc_norm_detrend(df):

    return (df.Adj_Detrend.diff()/df.Adj_Detrend.diff().mean())

## Detrend elevations in a radial profile
def detrend_transect(df):

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
        A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
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
    intitial_value = df[df.Distance == 0].Detrend.values[0]
    return df.Detrend - intitial_value

## Create a polynomial fit
def poly_fit(df, order):
    coeff = np.polyfit(df.Distance.values, df.Elevation.values, order)
    return np.polyval(coeff, df.Distance.values)

def set_initial_poly_point(df):
    intitial_value = df[df.Distance == 0].poly_detrend.values[0]
    return df.poly_detrend - intitial_value

# @jit(nopython = True)
## Checks if lists are same length
## Doesn't actually do anything since I am using a 1-D array as input
def _datacheck_peakdetect(x_axis, y_axis):
    '''Used in peakdetect(), and pulled from same source.
    Checks to make sure that x_axis and y_axis are arrays of 
    the same length.
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
       
    # check input data
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
    mn, mx = np.Inf, -np.Inf
    
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
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ###look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    # try:
    # if len(max_peaks) > 0:
    if dump[0]:
        max_peaks.pop(0)
    else:
        min_peaks.pop(0)
            # except IndexError:
            #no peaks were found, should the function return empty lists?
            # pass
        # Return array of indicies 
    return np.asarray([n[0] for n in max_peaks])
    # else:
    #     return np.asarray([-9999])

@jit(nopython = True)
## Get count of values within a range
## Returns the sum of True values within a boolean array. count() does not work here. sum() returns the count of True values 
def count_values_within_range(df, min_val, max_val):
    """Counts number of rows within a search window
    :param df: array of distance values - pd/gpd dataframe containing Distance column
    :param min_val: Minimum value of search window, int
    :param max_val: Maximum value of search window, int
    :return count: number of rows within search window 
    """
    # return len(df[(df.Distance.values >= min_val) & (df.Distance.values <= max_val)])

    # if c > 0:
    #     return 1
    # elif c == 0:
    #     return 0
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
    # try:
        # distances_array = df.Distance.values
        # intervals = pd.DataFrame({'Start': np.arange(0, distance, window_size), 'End': np.arange(window_size, distance + 1, window_size)})
    intervals_start = np.arange(0, distance, window_size)
    intervals_end = np.arange(window_size, distance + window_size, window_size)
    intervals_counts = np.asarray([count_values_within_range(distances_array, x, y) for x,y in zip(intervals_start, intervals_end)])
    intervals_counts = np.where(intervals_counts > 0, 1, 0) 
    intervals_angle = np.asarray(len(intervals_counts) * [angle]) 
    # except ValueError:
    #     # distances_array = df.Distance.values
    #     # intervals = pd.DataFrame({'Start': np.arange(0, distance, window_size), 'End': np.arange(window_size, distance + window_size, window_size)})
    #     intervals_start = np.arange(0, distance, window_size)
    #     intervals_end = np.arange(window_size, distance + window_size, window_size)
    #     intervals_counts = np.asarray([count_values_within_range(distances_array, x, y) for x,y in zip(intervals_start, intervals_end)])
    #     intervals_counts = np.where(intervals_counts > 0, 1, 0)
    #     intervals_angle = np.asarray(len(intervals_counts) * [angle])

    return intervals_start, intervals_end, intervals_counts, intervals_angle

## Round a number to the nearest 100
def roundup(x):

    return x if x % 100 == 0 else x + 100 - x % 100

## Create array containing detrended elevations and matching indexes
def set_up_peak_search(df, detrend_type):
    '''Creates a list of tuples containing arrays for the transect angle, detrended elevation, and index value
    :param df: pandas dataframe containing detrended elevations.
    :detrend_type: str, type of detrened elevation, 'Adj_Detrend' or 'Polynomial'
    :return : list of tuples
    '''
    interval_data = []
    angle_intervals = df.Angle.unique()
    if detrend_type == 'Adj_Detrend':
        for i in angle_intervals:
            interval_data.append(tuple([i, df[df.Angle == i].Adj_Detrend.values, np.asarray(range(len(df[df.Angle == i].Adj_Detrend.values)))]))
    elif detrend_type == 'Polynomial':
        for i in angle_intervals:
            interval_data.append(tuple([i, df[df.Angle == i].poly_detrend.values, np.asarray(range(len(df[df.Angle == i].Adj_Detrend.values)))]))

    return interval_data

## Search for peaks
# @jit(nopython = True)
def search_for_peaks(interval_data, r0, r1):
    '''used to search detrended elevation cross sections for peaks
    :param interval_data: list of tuples containing the detrended elevations and indexes for each transect angle
    :param r0: lookahead value
    :param r1: delta value
    :return maxp: returns list of peaks by [[index, value], [index, value]...]
    '''
    detected_peaks = []
    transect_angle = []
    n = 0
    for i in interval_data:
        try:
            maxp = peakdetect(y_axis = i[1], x_axis = i[2], lookahead = r0, delta= r1)
            detected_peaks.append(maxp)
            transect_angle.append(np.asarray(i[0]))
        except IndexError:
            logging.warning('Empty array: Angle {}; lookahead {}; delta {}'.format(n, r0, r1))

        n +=1
    
    return transect_angle, detected_peaks ## Returns the angle and index for the peak in two lists of arrays

## Not used, why is this here?
def unzip(data):
    inds, _ = zip(*data)
    return list(inds)

## Perform iterative search for local maxima
def iterative_peak_search(lines, max_lookahead, search_distance, window_size, p_count = 4):
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
    :return potential_ring_distances: list containing all identified peak distances across all iterations, not currently used once reported
    :return pd df: pandas dataframe containing all identified peaks detected in at least five transects during an iteration
    """
    potential_ring_distances = []
    peak_df = []
    param_array = np.array(np.meshgrid(np.arange(1, max_lookahead + 1,1), 0.1)).T.reshape(-1, 2)   
    print('Starting window: {}'.format(window_size))
    # print('Running with adjusted detrend values')
    interval_data = set_up_peak_search(lines, detrend_type = 'Adj_Detrend')

    for r in param_array:
        # print('Running Lookahead {}, Delta {}'.format(r[0],r[1]))
        # search for peaks by iterating over detrended elevations by angle
        transect_angles, max_peaks = search_for_peaks(interval_data, r0 = int(r[0]), r1 = int(r[1]))
        max_peaks = [(x, y) for x,y in zip(transect_angles, max_peaks)]
        potential_peaks = []
        # print(max_peaks)
        for p in max_peaks:
            # potential_peaks.append(lines[lines.Angle == p[0]].iloc[[n[0] for n in p[1]]])
            # try:
            #     print(p)
            #     if p[1] is not None:
            #         if p[1][0] != -9999:
            potential_peaks.append(lines[lines.Angle == p[0]].iloc[p[1]])
                    
            # except:
            #     if p is not None:
            #         logging.warning('Peakdetect reported no peaks.')
            #     else:
            #         print(p)
            #         sys.exit()
        potential_peaks = pd.concat(potential_peaks)

    
        # was exporting pandas dataframes
        # make sure it exports multiple lists of arrays
        # Each angle is a radial profile 
        all_start, all_end, all_counts, all_angles = zip(*[peaks_by_distance(potential_peaks[potential_peaks.Angle == ang].Distance.values, search_distance, window_size, angle = ang) for ang in potential_peaks.Angle.unique()])
        t = pd.DataFrame({'Start': np.concatenate(all_start), 
                          'End': np.concatenate(all_end), 
                          'Count': np.concatenate(all_counts), 
                          'Angle': np.concatenate(all_angles)})

        t = t.groupby(by = 'End')['Count'].sum().reset_index(name='Count').sort_values(by = 'Count', ascending = False).reset_index(drop=True)
        # Only select peaks that appear in more than 4 transects
        t = t[t.Count > 4].reset_index(drop = True)
        potential_ring_distances.append(np.repeat(t[t.Count > p_count].End.values, 2))

        ## Set up to export all peak counts
        t['Lookahead'] = len(t.Count.values) * [r[0]]
        t['Type'] = len(t.Count.values) * ['CAD']
        peak_df.append(t)

    # print('Running with polynomial detrend values')
    interval_data = set_up_peak_search(lines, detrend_type = 'Polynomial')

    for r in param_array:
        # print('Running Lookahead {}, Delta {}'.format(r[0],r[1]))
        # search for peaks by iterating over detrended elevations by angle
        transect_angles, max_peaks = search_for_peaks(interval_data, r0 = int(r[0]), r1 = int(r[1]))
        max_peaks = [(x, y) for x,y in zip(transect_angles, max_peaks)]
        potential_peaks = []
        for p in max_peaks:
            potential_peaks.append(lines[lines.Angle == p[0]].iloc[p[1]])
        potential_peaks = pd.concat(potential_peaks)
    
        
        ## Check with above ##
        all_start, all_end, all_counts, all_angles = zip(*[peaks_by_distance(potential_peaks[potential_peaks.Angle == i].Distance.values, search_distance, window_size, angle = i) for i in potential_peaks.Angle.unique()])
        t = pd.DataFrame({'Start': np.concatenate(all_start), 
                          'End': np.concatenate(all_end), 
                          'Count': np.concatenate(all_counts), 
                          'Angle': np.concatenate(all_angles)})        


        t = t.groupby(by = 'End')['Count'].sum().reset_index(name='Count').sort_values(by = 'Count', ascending = False).reset_index(drop=True)[:10]
        t = t[t.Count > 4].reset_index(drop = True)
        potential_ring_distances.append(t[t.Count > p_count].End.values)
        
        # Set up to export all peak counts
        t['Lookahead'] = len(t.Count.values) * [r[0]]
        t['Type'] = len(t.Count.values) * ['PND']
        peak_df.append(t)   

    peak_df = pd.concat(peak_df).reset_index(drop=True)
    peak_df['Window'] = len(peak_df) * [window_size]     

    # return np.ndarray.tolist(np.concatenate(potential_ring_distances)), pd.concat(test_df).reset_index(drop=True)
    return peak_df

## Find lowest count
def find_difference(df, val_a, val_b):
    """Given two distances, reports the one with the lowest frequency
    :param df: pd dataframe containing distances and frequencies
    :param a,b: distances
    :return distance: distance with lowest frequency
    """
    if (abs(val_a-val_b)) < 1000:
        mfp = min(df[df.Distance == val_a].Adj_Count.values[0], df[df.Distance == val_b].Adj_Count.values[0])
        return df[df.Adj_Count == mfp].Distance.values[0]
    
## Attempts to classify peaks as false positive or impact structure
def sort_potential_rings(pk_list, crater_type, freq_denominator, inner_cavity_estimate, iterate_mode = True):
    """Attemtps to select peaks that correspond to NEO impact structures
    :param pk_list: List of peak distances identified through the iterative peak search function
    :param crater_type: str, whether the crater has a 'simple' or 'complex' structure
    :param freq_denominator: total number of possible counts used to calculate frequency
    :param inner_cavity_estimate: Estimate of inner cavity radius in meters
    :param iterate_mode: bool, used to bypass ring labeling when using results from multiple iterations
    :return : pandas dataframe containing the radii of the most probable rings
    """
    logging.info('{} crater structure selected'.format(crater_type))
    p_rings = pd.DataFrame({'Distance': pk_list})
    possible_rings = p_rings.groupby(['Distance']).size().reset_index(name='counts').sort_values(by = 'counts',
                                                                                                 ascending = False).reset_index(drop = True)
    possible_rings['Frequency'] = possible_rings.counts/freq_denominator
    all_rings = possible_rings[possible_rings['Frequency'] >= 0.1]
    possible_rings = possible_rings[possible_rings['Frequency'] >= 0.33]
    possible_rings['Structure'] = ['False Positive'] * len(possible_rings)

    if iterate_mode is True:
        return possible_rings, all_rings
    else:
        ### Identifying inner depression ###
        if crater_type == 'complex':     
            # Inner Cavity is most frequent distance accounting for inner cavity estimate
            ic_diff = list(1 - abs(possible_rings[possible_rings.Distance < inner_cavity_estimate].Distance - inner_cavity_estimate)/inner_cavity_estimate)
            a = [(x,y)for x,y in zip(possible_rings[possible_rings.Distance < inner_cavity_estimate].Frequency, ic_diff)]

            # Check if most frequent with is actually the inner crater diameter. Select most similar to estimate if not.
            try:
                ## Old -- ID'd distance closest to estimate as the inner cavity
                if max(filter(lambda a: a[0] >= 0.5, a), key=itemgetter(1))[0] > inner_cavity_estimate/2:
                    possible_rings['Structure'] = np.where(possible_rings.Frequency == max(filter(lambda a: a[0] >= 0.5, a), key=itemgetter(1))[0],
                                                           'Inner Cavity', possible_rings.Structure)
                    logging.info('Inner Cavity Diameter Estimate: {}'.format(possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]))
                else:
                    possible_rings['Structure'] = np.where(possible_rings.Frequency == max(filter(lambda a: a[1] >= 0.5, a), key=itemgetter(1))[0],
                                                           'Inner Cavity', possible_rings.Structure)
                    logging.info('Inner Cavity Diameter Estimate: {}'.format(possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]))

            except ValueError:
                logging.warning('Unable to identify peaks within estimated diameter.')
                possible_rings['Structure'] = np.where(possible_rings.Frequency == np.nanmax(possible_rings.Frequency.values), 'Inner Cavity', possible_rings.Structure)            
            
            # Determine if there is a central peak using the cavity diameter
            # central_peak_distance = roundup(possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]*0.3)
            # logging.info('Determining if central peak is present.')
            # logging.info('Using {} m as search window'.format(central_peak_distance))
            # if possible_rings[possible_rings.Distance <= central_peak_distance].empty is False:
            #     # If there are multiple peaks within search distance, select peak with highest frequency
            #     possible_rings['Structure'] = np.where((possible_rings.Frequency == possible_rings[possible_rings.Distance <= central_peak_distance].Frequency.max()),
            #                                            'Central Peak', possible_rings.Structure)    
            
        elif crater_type == 'simple':
            # Simple crater will only have an inner cavity
            possible_rings['Structure'] = np.where(possible_rings.Frequency == possible_rings.Frequency.max(),
                                                   'Inner Cavity', possible_rings.Structure)
        
        ### Identify additional structures ###
        inner_radius = possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]
        if crater_type == 'complex':
            logging.info('Searching for additional rings using radius of {}'.format(inner_radius))
            maximum_distance = possible_rings.Distance.max()
            max_range = 0
            n = 1
            ring_distances = []
            if inner_radius > 5000:
                step = 2000
            else:
                step = 500
            # while max_range < maximum_distance:
            #     r = [1.8**0.5*inner_radius-500*n + inner_radius*n, 2.2**0.5*inner_radius+500*n + inner_radius*n]
            #     print(r)
            #     ring_distances.append(r)
            #     max_range = r[1]
            #     n +=1
            while max_range < maximum_distance:
                if n == 1:
                    r = [1.8**0.5*inner_radius-step*n + inner_radius*n, 2.2**0.5*inner_radius+step*n + inner_radius*n]
                    print(r)
                    ring_distances.append(r)
                    max_range = r[1]
                elif n != 1:
                    r = [max_range, 2.2**0.5*inner_radius+step*n + inner_radius*n]
                    print(r)
                    ring_distances.append(r)
                    max_range = r[1]
                    
                n +=1

            logging.info('Maximum number of potential rings: {}'.format(len(ring_distances)))
            for radii in ring_distances:
                possible_rings['Structure'] = np.where((possible_rings.Frequency == possible_rings[(possible_rings.Distance >= radii[0]) &
                                                                                                   (possible_rings.Distance <= radii[1])].Frequency.max()),
                                                       'Potential Ring', possible_rings.Structure)

            ## Removes any rings incorrectly identified within inner cavity
            possible_rings['Structure'] = np.where((possible_rings.Distance < inner_radius) &
                                                   (possible_rings.Structure == 'Potential Ring'),
                                                  'False Positive',
                                                  possible_rings.Structure)

            ## Removes rings if they are within 1000 m of each other by selecting the one with the highest frequency
            sub_pr = possible_rings[possible_rings.Structure == 'Potential Ring']
            sub_pr_dist = possible_rings[possible_rings.Structure == 'Potential Ring'].Distance.unique()
            false_peaks = [x for x in [find_difference(sub_pr,sub_pr_dist[i],sub_pr_dist[i+1]) for i in range(len(sub_pr_dist)-1)] if x is not None]
            
            if false_peaks:
                for d in false_peaks:
                    print('Removing distance {} due to proximity'.format(d))
                    possible_rings['Structure'] = np.where(possible_rings.Distance == d,'False Positive',possible_rings.Structure)

        elif crater_type == 'simple':
            logging.info('Searching for potential ejecta')
            possible_rings['Structure'] = np.where((possible_rings.Distance > inner_radius) &
                                                   (possible_rings.Distance < inner_radius * 3),
                                                   'Potential Ejecta', possible_rings.Structure)        
        return possible_rings, all_rings

## Get count of values within a range
## This could be switched to use @jit
def count_counts(df, min_val, max_val):
    """Counts number of rows within a search window
    :param df: pd/gpd dataframe containing Distance column
    :param min_val: Minimum value of search window, int
    :param max_val: Maximum value of search window, int
    :return count: number of rows within search window 
    """
    # return len(df[(df.Distance.values >= min_val) & (df.Distance.values <= max_val)])
    return df[((min_val <= df.Distance.values) & (df.Distance.values < max_val))].Count.count() ## Changed from .sum() 4/14/23

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
        intervals['Adj_Count'] = [count_counts(df, x, y) for x,y in zip(intervals.Start, intervals.End)]  
        return intervals
    except ValueError:
        intervals = pd.DataFrame({'Start': np.arange(0, distance, window_size), 'End': np.arange(window_size, distance + window_size, window_size)})
        intervals['Adj_Count'] = [count_counts(df, x, y) for x,y in zip(intervals.Start, intervals.End)]
        return intervals

## Estimates crater diameter using iteration results
## Not used at the moment 
def final_ring_sorting(df, inner_cavity_estimate, p_count):
    """Attempts to sort through all iteration results to identify the most probably crater diameters
    :param df: pandas dataframe containing the concatenated results from all peak detection iterations
    :param inner_cavity_estimate: Estimate of inner cavity radius in meters (args.d/2)
    :returns possible_rings: 
    """
    print(df[:10])
    df = df[df.Count > 4].reset_index(drop = True)
    df['Distance'] = df.End.values
    df = df[['Distance', 'Type', 'Count']]
    # Double count of CAD-detected peaks to weight it 2-1
    df = pd.concat([df, df[df.Type == 'CAD']])

    possible_rings = peaks_by_count(df, df.Distance.max(), 200)
    possible_rings = possible_rings.groupby(by = 'End')['Adj_Count'].sum().reset_index(name='Count').reset_index(drop=True)
    possible_rings = pd.DataFrame({'Distance': possible_rings.End.values, 'Adj_Count': possible_rings.Count.values})
    ic_diff = list(1 - abs(possible_rings[possible_rings.Distance < inner_cavity_estimate].Distance - inner_cavity_estimate)/inner_cavity_estimate)
    a = [(x,y)for x,y in zip(possible_rings[possible_rings.Distance < inner_cavity_estimate].Adj_Count, ic_diff)]
    possible_rings['Structure'] = len(possible_rings) * ['nan']
    logging.info('{}/{} matching windows had no peaks.'.format(len(possible_rings[possible_rings.Adj_Count == 0]), len(possible_rings.index)))
    possible_rings = possible_rings[possible_rings.Adj_Count != 0].reset_index(drop=True)

    ## New -- Selects most frequent distance within 1 km of inner_cavity_estimate
    upper_limit = inner_cavity_estimate + 500
    lower_limit = inner_cavity_estimate - 500
    possible_rings['Structure'] = np.where(possible_rings.Distance == possible_rings[(possible_rings.Distance <= upper_limit) &
                                        (possible_rings.Distance > lower_limit) &
                                        (possible_rings.Adj_Count == possible_rings[(possible_rings.Distance <= upper_limit) &
                                        (possible_rings.Distance > lower_limit)].Adj_Count.max())].Distance.values[0],
                                        'Inner Cavity', 'nan')
    maximum_distance = possible_rings.Distance.max()
    max_range = 0
    n = 1
    ring_distances = []
    try:
        inner_radius = possible_rings[possible_rings.Structure == 'Inner Cavity'].Distance.values[0]
    except IndexError:
        logging.warning('Something went wrong when identifying the inner cavity diameter. Using provided estimate instead: {}'.format(inner_cavity_estimate))
        logging.warning('You will need to do the diameter estimation separately.')
        print(possible_rings[possible_rings.Structure != 'nan'])
        inner_radius = inner_cavity_estimate

    if inner_radius > 5000:
        step = 2000
    else:
        step = 500

    logging.info('Inner Cavity Radius: {}'.format(inner_radius))

    while max_range < maximum_distance:
        if n == 1:
            r = [1.8**0.5*inner_radius-step*n + inner_radius*n, 2.2**0.5*inner_radius+step*n + inner_radius*n]
            print(r)
            ring_distances.append(r)
            max_range = r[1]
        elif n != 1:
            r = [max_range, 2.2**0.5*inner_radius+step*n + inner_radius*n]
            print(r)
            ring_distances.append(r)
            max_range = r[1]
        n +=1

    for radii in ring_distances:
        possible_rings['Structure'] = np.where((possible_rings.Adj_Count == possible_rings[(possible_rings.Distance >= radii[0]) &
                                                                            (possible_rings.Distance <= radii[1])].Adj_Count.max()) &
                                                                            (possible_rings.Distance >= radii[0]) & (possible_rings.Distance <= radii[1]),
                                                                            'Potential Ring', possible_rings.Structure)

    ## Removes any rings incorrectly identified within inner cavity
    possible_rings['Structure'] = np.where((possible_rings.Distance < inner_radius) &
                                           (possible_rings.Structure == 'Potential Ring'),
                                          'False Positive',
                                          possible_rings.Structure)

    ## Removes rings if they are within 1000 m of each other by selecting the one with the highest frequency
    sub_pr = possible_rings[possible_rings.Structure == 'Potential Ring']
    sub_pr_dist = possible_rings[possible_rings.Structure == 'Potential Ring'].Distance.unique()
    false_peaks = [x for x in [find_difference(sub_pr,sub_pr_dist[i],sub_pr_dist[i+1]) for i in range(len(sub_pr_dist)-1)] if x is not None]
    
    if false_peaks:
        for d in false_peaks:
            print('Removing distance {} due to proximity'.format(d))
            possible_rings['Structure'] = np.where(possible_rings.Distance == d,'False Positive',possible_rings.Structure)

    return possible_rings

## Estimates crater diameter using iteration results
def final_ring_sorting_v2(df, inner_cavity_estimate, p_count, count_threshold):
    """Attempts to sort through all iteration results to identify the most probably crater diameters
    :param df: pandas dataframe containing the concatenated results from all peak detection iterations
    :param inner_cavity_estimate: Estimate of inner cavity radius in meters (args.d/2)
    :param count_threshold: Value used to select potential outer rims. Count of detections must be within this value of the
                            probable outer rim to be considered part of the outer rim range.
    :returns possible_rings: Range of potential outer rims with their detected frequency
    """
    # print(df[:10])
    df = df[df.Count > 4].reset_index(drop = True)
    df['Distance'] = df.End.values
    df = df[['Distance', 'Type', 'Count']]
    # Double count of CAD-detected peaks to weight it 2-1
    df = pd.concat([df, df[df.Type == 'CAD']])

    if (args.mars):
        possible_rings = peaks_by_count(df, df.Distance.max(), 2000)
    else:
        possible_rings = peaks_by_count(df, df.Distance.max(), 200)
    possible_rings = possible_rings.groupby(by = 'End')['Adj_Count'].sum().reset_index(name='Count').reset_index(drop=True)
    possible_rings = pd.DataFrame({'Distance': possible_rings.End.values, 'Adj_Count': possible_rings.Count.values})
    ic_diff = list(1 - abs(possible_rings[possible_rings.Distance < inner_cavity_estimate].Distance - inner_cavity_estimate)/inner_cavity_estimate)
    a = [(x,y) for x,y in zip(possible_rings[possible_rings.Distance < inner_cavity_estimate].Adj_Count, ic_diff)]
    possible_rings['Structure'] = len(possible_rings) * ['nan']
    logging.info('{}/{} matching windows had no peaks.'.format(len(possible_rings[possible_rings.Adj_Count == 0]), len(possible_rings.index)))
    possible_rings = possible_rings[possible_rings.Adj_Count != 0].reset_index(drop=True)

    if (args.mars):
        upper_limit = inner_cavity_estimate + 2500
        lower_limit = inner_cavity_estimate - 5000
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

    # Find all IC diameters within 1 km and 100 count
    ### Changed from +/- 600 to +/- 500 on 8/14/23
    if (args.mars):
        possible_rings['Structure'] = np.where((possible_rings.Structure == 'nan') & 
                                               (possible_rings.Distance <= ic_radius + 6000) &
                                               (possible_rings.Distance >= ic_radius - 6000) & 
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
    if (args.mars):
        lower_OR_threshold = 1.5*ic_min + 15000
        upper_OR_threshold = 2.2*ic_max + 15000
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
    min_val = np.nanmin(val_list)
    max_val = np.nanmax(val_list)
    range_value = max_val - min_val
    return range_value

## Get count of NaN within list
def count_nan(val_list):
    return val_list.count(np.nan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=os.path.abspath, metavar='', required = True, 
                        help="""Path to DEM to be used for calculating ring diameters.""")
    parser.add_argument('-c', nargs='+', type=int, metavar='', required = True,
                        help="""Easting and Northing for crater center, e.g. '237740 2720414'.""")
    parser.add_argument('-t', type=str, metavar='', required = True, 
                        help="""Crater type, i.e., simple or complex.""")
    parser.add_argument('-d', type=int, metavar='', required = True,
                        help="""Loose estimate for inner cavity diameter in meters.
                        It is better to overestimate than underestimate this parameter.""")
    parser.add_argument('-w', type=int, metavar='', 
                        help="""Matching window size in meters.
                        Used to count the number of peaks.""")
    parser.add_argument('-pc', type=int, metavar='', 
                        help="""Number of detected peaks to be considered a potential outer rim.
                        Used to filter out individual peaks.""")
    parser.add_argument('-a', type=float, metavar='', required = True, 
                        help="""Step size in degrees to create radii. If unsure, set to 1.""")
    parser.add_argument('-o', type=os.path.abspath, metavar='', required = True,
                        help="""Path to output file with .shp extension.""")
    parser.add_argument('-iterate', action='store_true',
                        help="""Flag indicating the iteration of points around the provided center coordinates.""")    
    parser.add_argument('-sd', type=int, metavar='', 
                        help="""Search distance variant: 1-5.
                        1: Iterate through each search distance variant
                        2: roundup((args.d/2) * 2.2 + 2000)
                        3: roundup((args.d/2) * 3)
                        4: roundup((args.d/2) * 4)
                        5: roundup((args.d/2) * 5)
                        6: Iterate through n numbers of search distances. Currently set to 100.""")
    parser.add_argument('-mars', action='store_true',
                        help="""Flag to adjust matching window sizes to account for scaling difference on Mars.""") 
    parser.add_argument('-venus', action='store_true',
                        help="""Flag to sample using rasterio and not numpy.""") 
    args = parser.parse_args()
    timeStart = runTime.time()

    img = rasterio.open(args.i)
    geotransform = img.transform
    center_x, center_y = args.c[0], args.c[1]
    # calculate parameters for future use
    # search distance
    if args.sd == 1:
        search_distance_2 = roundup((args.d/2) * 2.2 + 2000)
        search_distance_3 = roundup((args.d/2) * 3)
        search_distance_4 = roundup((args.d/2) * 4)
        search_distance_5 = roundup((args.d/2) * 5)
        search_distances = [search_distance_5, search_distance_4, search_distance_3, search_distance_2]
        search_distances = sorted(search_distances, reverse = True)

    elif args.sd == 6:
        ll = int(roundup((args.d/2) * 2.2 + 2000))
        ul = int(roundup((args.d/2) * 5))
        num_samples = 100
        search_distances = sorted(random.sample(range(ll, ul), num_samples), reverse = True)

        # for i in range(len(search_distances)):
        #     search_distances[i] = roundup(search_distances[i])

        while len(set(search_distances)) != num_samples:
            num_off = num_samples - len(set(search_distances))
            new_nums = random.sample(range(ll, ul), num_off)
            for i in range(len(new_nums)):
                search_distances.append(new_nums[i])
        search_distances = sorted(search_distances, reverse = True)

        logging.info('Random search distances: {}'.format(search_distances))

    elif args.sd == 2:
        search_distance = roundup((args.d/2) * 2.2 + 2000)
    elif args.sd == 3:
        search_distance = roundup((args.d/2) * 3)
    elif args.sd == 4:
        search_distance = roundup((args.d/2) * 4)
    elif args.sd == 5:
        search_distance = roundup((args.d/2) * 5)                
    # maximum lookahead of 500 m based on cell size
    if geotransform[0] < 200:
        max_lookahead = 500/geotransform[0]
    elif geotransform[0] >= 200:
        max_lookahead = 25
    # Calculate value to use for frequency
    freq_denominator = max_lookahead * 3

    logging.info('Parameters for radial profile analysis:')
    logging.info('Center Easting: {}'.format(center_x))
    logging.info('Center Northing: {}'.format(center_y))
    logging.info('Search Distance Variant: {}'.format(args.sd))
    logging.info('DEM Resolution: {}'.format(geotransform[0]))
    logging.info('Maximum Lookahead: {}'.format(max_lookahead))
    logging.info('Matching Window Size {} m'.format(args.w))
    if (args.pc):
        logging.info('Peak Count Threshold: {} matches'.format(args.pc))
        p_count = args.pc
    else:
        logging.info('No peak count threshold specified. Defaulting to > 4 peaks.')
        p_count = 4
    logging.info('Making radii every {} degrees'.format(args.a))
    if (args.iterate):
        logging.info('Iteration of different center coordinates selected.')
    else:
        logging.info('Using single center coordinate for sampling.')

    ## Added option for 30 m resolution DEMs with a count threshold of 25 -- 2024-jul-16
    ## Added option for 2 m resolution DEMs with count thrshold 210, which will ensure the 2 m ArcticDEM identifies the 23 km ring at Haughton -- 2024-jul-16
    if geotransform[0] >= 50:
        # count_threshold = 10
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

    combined_rings = []

    centroid_variants = [(center_x , center_y),
                        (center_x - 100, center_y - 100), 
                        (center_x - 100, center_y + 100), 
                        (center_x + 100, center_y + 100), 
                        (center_x + 100, center_y - 100)]

    n = 1

    ## Not currently using this; will not run correctly ##
    if (args.iterate):

        for i in centroid_variants:
            logging.info('Running centroid variant {} of 5'.format(n))
            v_x, v_y = i 

            logging.info('Generating sample points.')
            lines = create_radial_profile(center_x = v_x, center_y = v_y, distance = search_distance, sample_distance = geotransform[0], angle_step = args.a)
            logging.info('Sampling generated points.')
            lines = sample_radial_profile(lines, img)


            logging.info('Calculating metrics to use for local maxima search.')
            lines['Detrend'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: nan_least_squares(lines)).values)
            lines['Adj_Detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: set_initial_point(lines)).values
            lines['poly_fit'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: poly_fit(lines, 4)).values)
            lines['poly_detrend'] = lines.Elevation - lines.poly_fit

            # Not used for search, but provided anyways
            # lines['slope'] = lines.groupby(by = 'Angle').apply(lambda lines: calc_slope(lines)).values
            # lines['slope'] = np.where(lines.slope.isna() == True, 0, lines.slope)
            # lines['norm_detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: calc_norm_detrend(lines)).values
            # lines['norm_detrend'] = np.where(lines.norm_detrend.isna() == True, 0, lines.norm_detrend)


            logging.info('Iterating through different window sizes.')
            for i in [100, 150, 200, 250, 300, 350]:
                logging.info('Starting search with window size: {}'.format(i))
                ppk = iterative_peak_search(lines, max_lookahead = max_lookahead, search_distance = search_distance, window_size = i, p_count = p_count)
                potential_rings, all_rings = sort_potential_rings(ppk, crater_type = 'complex', freq_denominator = freq_denominator, inner_cavity_estimate = args.d/2, iterate_mode = True)
                potential_rings['Window'] = len(potential_rings) * [i]
                all_rings['Window'] = len(all_rings) * [i]
                combined_all.append(all_rings)
                combined_rings.append(potential_rings)
            n+=1

    else:

        ## Main analysis ##
        if (args.sd == 1) | (args.sd == 6):
            logging.info('Generating sample points using maximum sampling distance of {}.'.format(search_distances[0]))
            lines = create_radial_profile(center_x = center_x, center_y = center_y, distance = search_distances[0], sample_distance = geotransform[0], angle_step = args.a)
            logging.info('Sampling generated points.')
            lines = sample_radial_profile(lines, img, venus = args.venus)
            lines = lines[lines.Elevation != -9999]

            for x in search_distances:
                logging.info('Starting Search Distance: {}'.format(x))
                lines = lines[lines.Distance <= x]
                logging.info('Calculating metrics to use for local maxima search.')
                # lines['Elevation'] = np.where(lines.Elevation < -1000, np.nan, lines.Elevation)
                lines['Detrend'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: nan_least_squares(lines)).values)
                lines['Adj_Detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: set_initial_point(lines)).values
                lines['poly_fit'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: poly_fit(lines, 4)).values)
                lines['poly_detrend'] = lines.Elevation - lines.poly_fit
                # lines['poly_detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: set_initial_poly_point(lines)).values
                # lines.to_csv('G:/Crater/mars/test_gale.csv')

                iteration_rings = []

                if args.mars == True:
                    swds = [1000, 1500, 2000, 2500, 3000, 3500]
                else:
                    swds = [100, 150, 200, 250, 300, 350]

                iteration_rings = Parallel(n_jobs = 3, verbose = 1)(delayed(iterative_peak_search)(lines = lines, 
                    max_lookahead = max_lookahead, 
                    search_distance = x, 
                    window_size = i, 
                    p_count = p_count) for i in swds)

                # for i in [100, 150, 200, 250, 300, 350]:
                #     logging.info('Starting search with window size: {}'.format(i))
                #     rings_df = iterative_peak_search(lines, max_lookahead = max_lookahead, search_distance = x, window_size = i, p_count = p_count)
                #     # potential_rings, all_rings = sort_potential_rings(ppk, crater_type = 'complex', freq_denominator = freq_denominator, inner_cavity_estimate = args.d/2, iterate_mode = True)
                #     # potential_rings['Window'] = len(potential_rings) * [i]
                #     # all_rings['Window'] = len(all_rings) * [i]
                #     rings_df['Window'] = len(rings_df) * [i]

                #     iteration_rings.append(rings_df)  

                iteration_rings = pd.concat(iteration_rings).reset_index(drop=True)  
                final_diameters_all = final_ring_sorting_v2(df = iteration_rings, inner_cavity_estimate = args.d/2, p_count = p_count, count_threshold = count_threshold)  
                final_diameters_all['Structure'] = np.where(final_diameters_all.Structure == 'nan', 'False Positive', final_diameters_all.Structure)  
                final_diameters_all.to_csv('{}_{}.csv'.format(args.o[:-4], x), index=False)
                combined_rings.append(final_diameters_all)

            logging.info('Combining results across each search distance.')
            logging.info('Beginning calculation of outer rim prediction interval.')
            
            outer_rim_results = []
            for results in combined_rings:
                a = results[results.Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
                outer_rim_results.append(a[(a.Structure != 'IC Range') & (a.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True))

            # a = combined_rings[0][combined_rings[0].Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # b = combined_rings[1][combined_rings[1].Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # c = combined_rings[2][combined_rings[2].Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # d = combined_rings[3][combined_rings[3].Structure != 'False Positive'].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # a_OR = a[(a.Structure != 'IC Range') & (a.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # b_OR = b[(b.Structure != 'IC Range') & (b.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # c_OR = c[(c.Structure != 'IC Range') & (c.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)
            # d_OR = d[(d.Structure != 'IC Range') & (d.Structure != 'Inner Cavity')].sort_values(by = 'Adj_Count', ascending = False).reset_index(drop=True)

            u_dist= []
            # for i in [a_OR,b_OR,c_OR,d_OR]:
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
            df.to_csv('{}_OR_Interval.csv'.format(args.o[:-4]), index=False)

        ## Not currently using this ##
        else:
            logging.info('Generating sample points.')
            lines = create_radial_profile(center_x = center_x, center_y = center_y, distance = search_distance, sample_distance = geotransform[0], angle_step = args.a)
            logging.info('Sampling generated points.')
            lines = sample_radial_profile(lines, img)
            lines = lines[lines.Elevation != -9999]
            logging.info('Calculating metrics to use for local maxima search.')
            # lines['Elevation'] = np.where(lines.Elevation < -1000, np.nan, lines.Elevation)
            lines['Detrend'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: nan_least_squares(lines)).values)
            lines['Adj_Detrend'] = lines.groupby(by = 'Angle').apply(lambda lines: set_initial_point(lines)).values
            lines['poly_fit'] = np.concatenate(lines.groupby(by = 'Angle').apply(lambda lines: poly_fit(lines, 4)).values)
            lines['poly_detrend'] = lines.Elevation - lines.poly_fit
            # logging.info('Writing points')
            # lines.to_csv('{}_all_points.csv'.format(args.o[:-4]), index=False)
            # logging.info('Points Written')

            for i in [100, 150, 200, 250, 300, 350]:
                logging.info('Starting search with window size: {}'.format(i))
                ppk, test_df = iterative_peak_search(lines, max_lookahead = max_lookahead, search_distance = search_distance, window_size = i, p_count = p_count)
                potential_rings, all_rings = sort_potential_rings(ppk, crater_type = 'complex', freq_denominator = freq_denominator, inner_cavity_estimate = args.d/2, iterate_mode = True)
                potential_rings['Window'] = len(potential_rings) * [i]
                all_rings['Window'] = len(all_rings) * [i]
                test_df['Window'] = len(test_df) * [i]

                combined_all.append(all_rings)
                combined_rings.append(potential_rings)
                test_all.append(test_df)        

            combined_all = pd.concat(combined_all)
            combined_rings = pd.concat(combined_rings)
            test_all = pd.concat(test_all).reset_index(drop=True)            


    timeEnd = runTime.time()
    timeElapsedTotal = timeEnd - timeStart
    timeElapsedMin = np.floor(timeElapsedTotal / 60)
    timeElapsedSec = timeElapsedTotal % 60

    # Print completion message
    print('Script Completed in %d min %d sec.' % (timeElapsedMin, timeElapsedSec))
    print('\n')

    
    ## This does not function, but it is not important. Will fix when I have time. 
    # rings = []
    # for i in df.Diameter:
    # ## Create circle
    #     pt = gpd.GeoDataFrame(pd.DataFrame({'Diameter': [i]}), geometry = [Point((center_x, center_y))])
    #     pt['geometry'] = pt.geometry.buffer(i/2)
    #     rings.append(pt)

    # rings = pd.concat(rings)
    # rings = rings.set_crs(img.crs.to_epsg())
    # rings = rings.merge(df, on = 'Diameter')
    # rings.to_file(args.o)





