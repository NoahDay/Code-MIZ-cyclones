import numpy as np
import xarray as xr

def getClosestPoint1D(array, target):
    idx = np.argmin(np.abs(array - target))
    return array[idx], idx

def getClosestPoint2D(grid_x, grid_y, target):
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    distances = np.linalg.norm(grid_points - target, axis=1)
    closest_index = np.argmin(distances)
    closest_coord = grid_points[closest_index]
    closest_2d_index = np.unravel_index(closest_index, grid_x.shape)
    return closest_coord, closest_2d_index

def timeDerivative(ds, var):
    # Hourly data
    print('Warning [timeDerivative]: Assuming hourly data then do derivative per model timestep')
    dt = 1#/24
    tmp_var = var[1:-2]
    data_var = ds[tmp_var]
    time_derivative = data_var.diff(dim='time') / dt
    time_derivative = time_derivative.assign_coords(time=ds['time'].isel(time=slice(1, None)))
    time_derivative.name = var
    ds[var] = time_derivative
    ds[var].attrs['long_name'] = "change in ice area (aggregate) per model timestep"
    return ds