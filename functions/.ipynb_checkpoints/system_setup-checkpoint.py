import platform
from io import BytesIO
import sys

def getSavePath():
    if platform.system() == "Darwin":
        sys.path.insert(0, '/Users/noahday/GitHub/Code-MIZ-cyclones/functions')
        savepath = '/Users/noahday/GitHub/Code-MIZ-cyclones/figures/'
        netcdfpath = '/Volumes/NoahDay1TB/Code-MIZ-cyclones/'
        cyclone_filepath = '/Users/noahday/GitHub/Code-MIZ-cyclones/data/ERA-tracks/ERA-Interim_cyclone_tracks_1979-2018_south_of_60S.csv'
    elif platform.system() == "Linux":
        sys.path.insert(0, '/home/566/nd0349/Code-MIZ-cyclones/functions')
        savepath = '/home/566/nd0349/Code-MIZ-cyclones/figures/'
        cyclone_filepath = '/home/566/nd0349/Code-MIZ-cyclones/data/ERA-tracks/ERA-Interim_cyclone_tracks_1979-2018_south_of_60S.csv'
        netcdfpath = '/g/data/gv90/nd0349/Code-MIZ-cyclones/'
    else:
        print('ERROR: OS not found')
    return savepath, netcdfpath