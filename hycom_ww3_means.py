from pts_parcels import _get_indices_from_landmask
from processing import get_iot_lon_lat_range
from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time
from tools import log
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import os

def calculate_mean_current_field(months, lon_range, lat_range, input_dir):
    u_all = []
    v_all = []
    for month in months:
        start_date = datetime(2008, month, 1)
        end_date = add_month_to_time(start_date, 1)
        n_days = (end_date-start_date).days
        for i in range(n_days):
            date = start_date+timedelta(days=i)
            input_path = f'{input_dir}{date.strftime("%Y%m%d")}.nc'
            log.info(f'Reading data from: {input_path}')
            lon, lat, u, v = _read_current_data(input_path, lon_range, lat_range)
            u_all.append(u)
            v_all.append(v)
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    log.info(f'Calculating mean u and v')
    u_mean = np.nanmean(u_all, axis=0)
    v_mean = np.nanmean(v_all, axis=0)
    return lon, lat, u_mean, v_mean

def _write_mean_current_field_to_netcdf(lon, lat, u, v, output_path):
    log.info(f'Writing output to netcdf file: {output_path}')
    nc = Dataset(output_path,'w', format='NETCDF4')
    # define dimensions
    nc.createDimension('lat', len(lat))        
    nc.createDimension('lon',len(lon))
    # define variables
    nc_lon = nc.createVariable('lon', float, 'lon', zlib=True)
    nc_lat = nc.createVariable('lat', float, 'lat', zlib=True)
    nc_u = nc.createVariable('u', float, ('lat', 'lon'), zlib=True)
    nc_v = nc.createVariable('v', float, ('lat', 'lon'), zlib=True)
    # write variables
    nc_lon[:] = lon
    nc_lat[:] = lat
    nc_u[:] = u
    nc_v[:] = v
    nc.close()

def _read_current_data(input_path, lon_range, lat_range):
    indices = _get_indices_from_landmask(lon_range, lat_range, input_path=input_path)
    netcdf = Dataset(input_path)
    lon = netcdf['lon'][indices['lon']].filled(fill_value=np.nan)
    lat = netcdf['lat'][indices['lat']].filled(fill_value=np.nan)
    u = netcdf['u'][0, :, :][indices['lat'], :][:, indices['lon']].filled(fill_value=np.nan)
    v = netcdf['v'][0, :, :][indices['lat'], :][:, indices['lon']].filled(fill_value=np.nan)
    return lon, lat, u, v

if __name__ == '__main__':
    months = np.arange(1, 13)
    lon_range, lat_range = get_iot_lon_lat_range()
    
    # input_dir = get_dir_from_json('hycom_input')
    # output_dir = get_dir_from_json('hycom_means')
    input_dir = get_dir_from_json('ww3_input')
    output_dir = get_dir_from_json('ww3_means')
    
    for month in months:
        output_path = f'{output_dir}iot_{datetime(2008, month, 1).strftime("%b")}.nc'
        if os.path.exists(output_path):
            log.info(f'Mean file already exists, skipping: {output_path}')
            continue
    
        lon, lat, u, v = calculate_mean_current_field([month], lon_range, lat_range, input_dir)
        _write_mean_current_field_to_netcdf(lon, lat, u, v, output_path)
