# Plotting
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as colors
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cft
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from matplotlib import rc
import seaborn as sns
import matplotlib.transforms as mtransforms
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import matplotlib.transforms as mtransforms
import seaborn as sns
import matplotlib.path as mpath
import matplotlib.ticker as mticker
# from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
#                                 LatitudeLocator)
import cartopy.feature as cfea
import numpy as np
import os
from scipy.stats import pearsonr

import sys
import plot_settings
import pandas as pd


def create_map_axis(ax, LN, LT, draw_labels="off", textcolor = 'black'):
    COLOR_LAND = (0.7, 0.7, 0.7)
#    fig.subplots_adjust(bottom=0.05, top=0.95,
#                    left=0.04, right=0.95, wspace=0.02)
    # Limit the map to -60 degrees latitude and below.
    coord_lims = [-180, 180, -90, -50]
    ax.set_extent(coord_lims, ccrs.PlateCarree())
    
    if draw_labels != "none":
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.25, color='gray', alpha=0.5, linestyle='--', zorder=4)
        gl.top_labels = False
        gl.bottom_labels = False
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
        if draw_labels == "right": # right
            gl2 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.0, linestyle='--')
            gl2.xlocator = mticker.FixedLocator([180, 60, 120, 0])
            gl2.ylocator = mticker.FixedLocator([0])
        elif draw_labels == "left": # left
            gl2 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.0, linestyle='--')
            gl2.xlocator = mticker.FixedLocator([-180, -60, -120, 0])
            gl2.ylocator = mticker.FixedLocator([0])
        elif draw_labels == "middle":
            gl2 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.0, linestyle='--')
            gl2.xlocator = mticker.FixedLocator([-180, 0])
            gl2.ylocator = mticker.FixedLocator([0])
        elif draw_labels == "all":
            gl2 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.0, linestyle='--')
            gl2.xlocator = mticker.FixedLocator([-180, -60, -120, 0, 60, 120])
            gl2.ylocator = mticker.FixedLocator([0])
            
    #    gl.right_labels = False
    #    gl.left_labels = False
        
        gl.ylocator = mticker.FixedLocator([-80, -70, -60])
        gl2.ylocator = mticker.FixedLocator([0])
        #gl.ylocator = LatitudeLocator()
        gl2.xformatter = LongitudeFormatter()
        gl2.yformatter = LatitudeFormatter()
        gl2.xlabel_style = {'color': textcolor, 'weight': 'bold', 'size': 8, 
                          }
    #    gl.ylabel_style = {'size': 8}
    #    gl2.ylabel_style = {'size': 8}
    
    #    ax.add_feature(cfeature.LAND,color='gray')
    #    ax.add_feature(cfeature.OCEAN)

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    
#    cmap_mask = mpl.colors.ListedColormap([COLOR_LAND])
    # ax.pcolormesh(LN, LT, land_mask,
    #         transform=ccrs.PlateCarree(),
    #         cmap=cmap_mask,
    #         alpha = 1.0,
    #         shading='auto')
    land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                        edgecolor='black', facecolor=COLOR_LAND, linewidth=0.5)
    ax.add_feature(land_50m)
    
    ax.plot([180, 180], [-90, -84.6],
                     zorder=3, color=COLOR_LAND,
                     linewidth=0.75,
                     transform=ccrs.PlateCarree())

    
    return ax


def plot_style(plot_type="paper"):
    font_size = 22
    if plot_style == "paper":
        plt.style.use("ggplot")
        rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
        rc('text', usetex=True)
        font_size = 14
        mpl.rcParams.update({'font.size': 22})

        mpl.rc('xtick', labelsize=16) 
        mpl.rc('ytick', labelsize=16) 
        plt.style.use("ggplot")
        mpl.rcParams["axes.edgecolor"] = [0.6, 0.6, 0.6]
        mpl.rcParams["axes.linewidth"] = 1.0
    elif plot_style == "talk":
        font_size = 22
        sns.set_context("talk")
        plt.style.use("fivethirtyeight")
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        mpl.rcParams.update({'font.size': 22})

        mpl.rc('xtick', labelsize=16) 
        mpl.rc('ytick', labelsize=16) 

        fig = plt.figure(figsize=(13.33,7.5), dpi=96)
        plt.style.use("fivethirtyeight")
        mpl.rcParams["axes.edgecolor"] = [0.6, 0.6, 0.6]
        mpl.rcParams["axes.linewidth"] = 1.0
    return font_size


def add_subplot_label(fig, label, fontsize='large'):
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
           fontsize=fontsize, verticalalignment='top', fontfamily='serif',
           bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
def set_ax_date(ax, plot_type="ts"):
    # MONTHLY CLIMATOLOGY
    if plot_type == "clim":
        month_year_formatter = mdates.DateFormatter('%b') 
        monthly_locator = mdates.MonthLocator()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    # elif plot_tyle = "nolabel":
    #     month_year_formatter = mdates.DateFormatter('%b') 
    #     monthly_locator = mdates.MonthLocator()
    #     ax.xaxis.set_major_locator(mdates.MonthLocator())
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    # elif plot_type = "month":
    #     month_year_formatter = mdates.DateFormatter('%b') 
    #     monthly_locator = mdates.MonthLocator()
    #     ax.xaxis.set_major_locator(mdates.MonthLocator())
    #     ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    else:
        # MONTHLY WITH YEAR
        monthly_locator = mdates.MonthLocator()
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.set_xlabel('')
    
    
def plotSectorMap(df_single_track, savepath, npoints=100, fixed_lat=-40):
    track_number = df_single_track['# track'].values[0]
    lon_min, lon_max = df_single_track.longitude.min(), df_single_track.longitude.max()
    if (np.abs(df_single_track.longitude.min() - df_single_track.longitude.max()) > 180):
        lon_max = lon_max - 360
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': ccrs.SouthPolarStereo()})
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    min_lon = df_single_track.longitude.min()
    max_lon = df_single_track.longitude.max()
    if abs(min_lon - max_lon) > 180:
        over_seam = True
        plot_lons = df_single_track.longitude.apply(lambda x: x - 360 if x > 180 else x)
        min_lon, max_lon = plot_lons.min(), plot_lons.max()
    else: 
        plot_lons = df_single_track.longitude
        
    lons_smooth = np.linspace(min_lon, max_lon, npoints)
    latitudes = np.linspace(-90, fixed_lat, npoints)
    for lat in latitudes:
        ax.plot(lons_smooth, [lat] * len(lons_smooth), transform=ccrs.PlateCarree(), color='red', linewidth=0.5, alpha=0.5)
    
    ax.plot(plot_lons, df_single_track.latitude, transform=ccrs.PlateCarree())
    #ax.plot([lon_min, lon_min], [-90, fixed_lat], transform=ccrs.PlateCarree(), color='red', linewidth=2)
    #ax.plot([lon_max, lon_max], [-90, fixed_lat], transform=ccrs.PlateCarree(), color='red', linewidth=2)
    #ax.plot(np.linspace(lon_min, 100, npoints), np.full(npoints, fixed_lat), transform=ccrs.PlateCarree(), color='red', linewidth=2)
    ax.set_extent([-180, 180, -90, fixed_lat+2], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.set_title('Track number: {}'.format(str(track_number)))
    plt.tight_layout(pad=0.05)
    # Define the directory path
    directory = savepath + "gif/{}".format(track_number)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(savepath + "gif/{}/{}_sector_map.png".format(track_number,track_number), 
                dpi = 300, transparent=True, bbox_inches='tight')
    plt.show()
    return ax
    
def addGridLabels(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Suppress top labels
    gl.right_labels = False  # Suppress right labels
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}


def annotate_correlation(data, var1, var2, **kws):
    r, _ = pearsonr(data[var1], data[var2])
    ax = plt.gca()
    ax.text(1-plot_settings.label_x, plot_settings.label_y, f'$r$ = {r:.2f}', transform=ax.transAxes, 
            fontsize=plot_settings.titlesize, verticalalignment='top', ha='right')

    

def testPlot():
    # Create sample data for plotting
    np.random.seed(42)
    data = pd.DataFrame({
        'var1': np.random.rand(100),
        'var2': np.random.rand(100) + np.random.rand(100) * 0.5  # Slightly correlated data
    })

    fig, ax = plt.subplots(figsize=(plot_settings.width, plot_settings.height))
    ax.scatter(data['var1'], data['var2'])
    annotate_correlation(data, 'var1', 'var2')
    ax.xaxis.set_major_formatter(plot_settings.formatter)
    ax.yaxis.set_major_formatter(plot_settings.formatter)
    ax.set_xlabel('X-label')
    ax.set_ylabel('Y-label')
    plt.show()


