from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
from simple_maths import getClosestPoint1D, getClosestPoint2D


def convertFilenameToDatetime(filename, model_timestep='h'):
    if model_timestep == 'h':
        date_time_str = filename.split('.')[1]
        date_str, seconds_str = date_time_str[:10], date_time_str[11:]
        date_part = datetime.strptime(date_str, "%Y-%m-%d")
        seconds_part = timedelta(seconds=int(seconds_str))
        output = date_part + seconds_part
    elif model_timestep == 'd':
        date_time_str = filename.split('.')[1]
        date_str, seconds_str = date_time_str[:10], date_time_str[11:]
        date_part = datetime.strptime(date_str, "%Y-%m-%d")
        output = date_part 
    return output


def getCycloneDatetime(df_single_track):
    '''
    Convert the times in df_single_track to pandas datetimes.
    '''
    timesteps,cols = df_single_track.shape
    dates = []
    for idx in range(timesteps):
        temp_date = datetime(df_single_track['year'].values[idx], df_single_track['month'].values[idx], df_single_track['day'].values[idx], 
                             df_single_track['hour'].values[idx])
        dates.append(temp_date)
        
    dates = pd.to_datetime(dates)
    return dates

def getLongtiudeIndices(df_single_track, ds_CICE, buffer, dataset="CICE"):
    '''
    Given the coordinate range from df_single_track, return the indicies which cover this subdomain in ds_CICE.
    '''
    min_lon, max_lon = df_single_track.longitude.min(), df_single_track.longitude.max()
    if abs(min_lon - max_lon) > 270:
        over_seam = True
        temp_lons = df_single_track.longitude.apply(lambda x: x - 360 if x > 270 else x)
        min_lon, max_lon = temp_lons.min(), temp_lons.max()
    else:
        over_seam = False
    _, min_lon_idx = getClosestPoint1D(np.array(df_single_track.longitude.values), min_lon)
    min_lon_lat = df_single_track.latitude.values[min_lon_idx]
    _, max_lon_idx = getClosestPoint1D(np.array(df_single_track.longitude.values), max_lon)
    max_lon_lat = df_single_track.latitude.values[max_lon_idx]
    
    if dataset == "CICE":
        if over_seam:
            target = np.array([min_lon+360, min_lon_lat])
        else:
            target = np.array([min_lon, min_lon_lat])
        closest_coord_min_lon, closest_2d_index_min_lon = getClosestPoint2D(ds_CICE.TLON.values, ds_CICE.TLAT.values, target)
        target = np.array([max_lon, max_lon_lat])
        closest_coord_max_lon, closest_2d_index_max_lon = getClosestPoint2D(ds_CICE.TLON.values, ds_CICE.TLAT.values, target)
        if abs(closest_2d_index_min_lon[1] - closest_2d_index_max_lon[1]) > 700:
            range_1 = np.arange(closest_2d_index_min_lon[1] - buffer, ds_CICE.ni.values.max(), 1)
            range_2 = np.arange(ds_CICE.ni.values.min(), closest_2d_index_max_lon[1] + buffer, 1)
            ni_range = np.concatenate((range_1, range_2))
        else:
            ni_range = np.arange(closest_2d_index_min_lon[1] - buffer, closest_2d_index_max_lon[1] + buffer, 1)
        ni_range

    elif dataset == "jra55":
        if over_seam:
            target = np.array([min_lon+360])
        else:
            target = np.array([min_lon])
        
        closest_coord_min_lon, closest_1d_index_min_lon = getClosestPoint1D(ds_CICE.lon.values, target)
        target = np.array([max_lon])
        closest_coord_max_lon, closest_1d_index_max_lon = getClosestPoint1D(ds_CICE.lon.values, target)
        if closest_1d_index_min_lon < closest_1d_index_max_lon:
            print('True')
            ni_range = np.arange(closest_1d_index_min_lon, closest_1d_index_max_lon)
        else:
            ni_range = np.arange(closest_1d_index_max_lon, closest_1d_index_min_lon)

    return ni_range

def fixCiceDatetimes(ds_in, model_timestep='h'):
    timestamps = ds_in.time.values
    dt_index = pd.DatetimeIndex(timestamps)
    # Subtract one hour
    if model_timestep == 'h':
        dt_index_minus_1h = dt_index - pd.Timedelta(hours=1)
        ds_out = ds_in.copy()
#    ds_out.time.values = dt_index_minus_1h.to_numpy()
        ds_out = ds_out.assign_coords(time=dt_index_minus_1h)
    elif model_timestep == 'd':
        dt_index_minus_12h = dt_index - pd.Timedelta(hours=12)
        ds_out = ds_in.copy()
        ds_out = ds_out.assign_coords(time=dt_index_minus_12h)
    return ds_out

def subdomainCiceAndJra55(df_single_track, ds_CICE, ds_jra55, buffer=30):
    ni_range = getLongtiudeIndices(df_single_track, ds_CICE, buffer)
    ds_sel = ds_CICE.sel(ni=ni_range)
    
    lon_min, lon_max = df_single_track.longitude.min(), df_single_track.longitude.max()
    if abs(lon_min - lon_max) > 270:
        over_seam = True
        temp_lons = df_single_track.longitude.apply(lambda x: x - 360 if x > 270 else x)
        lon_min, lon_max = temp_lons.min(), temp_lons.max()
        ds_sel_jra55 = xr.concat([ds_jra55.sel(lon=slice(lon_min + 360 - buffer, 360)), ds_jra55.sel(lon=slice(0, lon_max + buffer))], dim='lon')
    else:
        ds_sel_jra55 = ds_jra55.sel(lon=slice(lon_min, lon_max))
    
    return ds_sel, ds_sel_jra55


def changeInSIC(ds_CICE_in):
    ds_CICE_tmp = ds_CICE_in.copy()
    ds_CICE_tmp['daidt'] = ds_CICE_tmp['daidtd'] 
    ds_CICE_tmp['daidt'].values += ds_CICE_tmp['daidtt'].values
    ds_CICE_tmp['daidt'].attrs['long_name'] = ds_CICE_tmp['daidtd'].attrs['long_name'][:-8] + 'both'
    return ds_CICE_tmp


def fixCyclicPoint(data, datatype='jra'):
    if datatype=='jra':
        if abs(np.diff(data.longitude, 1)).max() > 180:
            temp_lons = data.longitude.apply(lambda x: x - 360 if x > 180 else x)
            data.longitude = temp_lons
        else:
            temp_lons = data.longitude
    elif datatype=='CICE_lon':
        if abs(abs(np.diff(data.lon.values,1))).max() > 180:
            temp_lons = data.lon.values
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data.lon.values
    elif datatype=='CICE':
        if abs(abs(np.diff(data.TLON.values,1))).max() > 180:
            temp_lons = data.TLON.values
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data.TLON.values
    else:
        if abs(abs(np.diff(data,1))).max() > 180:
            temp_lons = data
            idx = temp_lons > 180
            temp_lons[idx] = temp_lons[idx] - 360
        else:
            temp_lons = data

    return temp_lons