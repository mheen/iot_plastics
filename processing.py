from tools.land import get_islands_from_toml, get_island_boxes_from_toml
from plastic_sources import get_iot_sources
from plot_tools.interactive_tools import plot_cycler
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import plot_box
from tools.ocean import Grid
from tools.files import get_dir_from_json
from tools.coordinates import get_index_closest_point
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import pandas as pd
from tools import log
import os

def get_iot_lon_lat_range():
    lon_range = [90., 128.]
    lat_range = [-20., 20.]
    return (lon_range, lat_range)

def process_parcels_netcdf(input_path:str, output_path:str):
    ds = xr.load_dataset(input_path)
    
    # create single time array for all particles
    dt = np.nanmax(np.diff(ds.time))
    tmin = np.nanmin(ds.time.values[ds.time != pd.NaT])
    tmax = np.nanmax(ds.time.values[ds.time != pd.NaT])
    time = np.arange(tmin, tmax+dt, dt)
    
    time_all = ds.time.values
    lon_org = ds.lon.values
    lat_org = ds.lat.values
    beached_org = ds.beached.values
    
    pid = ds.trajectory[:, 0].values
    
    # change particle lon and lat to fit single time array
    lon = np.empty((len(pid), len(time)))*np.nan
    lat = np.empty((len(pid), len(time)))*np.nan
    beached = np.empty((len(pid), len(time)))*np.nan
    for t in range(len(time)):
        i, j = np.where(time_all == time[t])            
        lon[i, np.repeat(t, len(i))] = lon_org[i, j]
        lat[i, np.repeat(t, len(i))] = lat_org[i, j]
        beached[i, np.repeat(t, len(i))] = beached_org[i, j]
    
    # write output
    ds_new = xr.Dataset(
        data_vars=dict(
            lon=(['pid', 'time'], lon),
            lat=(['pid', 'time'], lat),
            beached=(['pid', 'time'], beached)
        ),
        coords=dict(
            pid=('pid', pid),
            time=('time', time)
        )
    )
    ds_new.to_netcdf(output_path)
    log.info(f'Wrote processed parcels file to: {output_path}')

def process_particle_density(input_path:str, output_path:str,
                             lon_range=None, lat_range=None, dx=0.2):
    ds = xr.load_dataset(input_path)
    if lon_range == None:
        lon_range, _ = get_iot_lon_lat_range()
    if lat_range == None:
        _, lat_range = get_iot_lon_lat_range()
    grid = Grid(dx, lon_range, lat_range)
    
    total_particles = np.zeros((len(ds.time)))
    density = np.zeros((len(ds.time), grid.lat_size, grid.lon_size))
    shape_2d_density = density[0, :, :].shape
    
    lon_index, lat_index = grid.get_index(ds.lon.values, ds.lat.values)
    for t in range(len(ds.time)):
        density_1d = density[t, :, :].flatten()
        l_nonan = np.logical_and(~np.isnan(lon_index[:, t]), ~np.isnan(lat_index[:, t]))
        x = (lon_index[l_nonan, t]).astype('int')
        y = (lat_index[l_nonan, t]).astype('int')
        index_1d = np.ravel_multi_index(np.array([y, x]), shape_2d_density)
        np.add.at(density_1d, index_1d, 1)
        density[t, :, :] = density_1d.reshape(shape_2d_density)
        total_particles[t] += sum((~np.isnan(ds.lon[:, :t+1])).any(axis=1))
    
    # write output
    ds_new = xr.Dataset(
        data_vars=dict(
            density=(['time', 'lat', 'lon'], density),
            total_particles=(['time'], total_particles)
        ),
        coords=dict(
            time=('time', ds.time.values),
            lon=('lon', grid.lon),
            lat=('lat', grid.lat)
        )
    )
    ds_new.to_netcdf(output_path)
    log.info(f'Wrote processed parcels file to: {output_path}')

def _get_particle_release_time_and_pid_indices(ds:xr.Dataset) -> tuple[np.ndarray[int], np.ndarray[int]]:
    p_no_nan, t_no_nan = np.where(~np.isnan(ds.lon.values))
    p_first, i_sort = np.unique(p_no_nan, return_index=True)
    t_release = t_no_nan[i_sort]
    return p_first, t_release

def get_initial_particle_lon_lat(ds:xr.Dataset):
    i_all, j_all = np.where(~np.isnan(ds.lon.values))
    i_first, i_sort = np.unique(i_all, return_index=True)
    j_first = j_all[i_sort]
    lon = ds.lon.values[i_first, j_first]
    lat = ds.lat.values[i_first, j_first]
    return lon,lat

def particles_plot_cycler(ds:xr.Dataset, t_interval=1):
    
    lon_range, lat_range = get_iot_lon_lat_range()
    
    cki, ci = get_islands_from_toml()
    
    cki_box, ci_box = get_island_boxes_from_toml()
    l_box_cki = get_l_particles_in_box(ds, cki_box, any=False)
    l_box_ci = get_l_particles_in_box(ds, ci_box, any=False)
    l_box = np.logical_or(l_box_cki, l_box_ci)
    
    def single_plot(fig, req_time):
        # time index to plot
        t = list(ds.time.values).index(ds.sel(time=req_time, method='nearest').time.values)
        ds_t = ds.isel(time=t)
        
        l_beached = ds_t.beached == 1
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range, lat_range)
        ax.scatter(ds_t.lon.values[~l_beached], ds_t.lat.values[~l_beached], facecolor='k', edgecolor='none')
        ax.scatter(ds_t.lon.values[l_beached], ds_t.lat.values[l_beached], facecolor='#cc0000', edgecolor='none')
        ax.scatter(ds_t.lon.values[l_box[:, t]], ds_t.lat.values[l_box[:, t]], facecolor='#0000cc', edgecolor='none')
        ax.set_title(pd.to_datetime(ds_t.time.values).strftime('%d-%m-%Y %H:%M'))
        
        plot_box(ax, cki['lon_range'], cki['lat_range'])
        plot_box(ax, ci['lon_range'], ci['lat_range'])
        
    t = np.arange(0, len(ds.time), t_interval)
    time = ds.time.values[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def get_l_particles_in_box(ds:xr.Dataset, box:dict, any=True, beached=False) -> np.ndarray[bool]:
    '''
    Finds particles that have been in a box (defined by lon_range and lat_range list)
    at any time.
    '''
    l_lon = np.logical_and(box['lon_range'][0] <= ds.lon.values, ds.lon.values <= box['lon_range'][1])
    l_lat = np.logical_and(box['lat_range'][0] <= ds.lat.values, ds.lat.values <= box['lat_range'][1])
    
    if beached == True:
        l_lon = np.logical_and(l_lon, ds.beached.values == 1)
        l_lat = np.logical_and(l_lat, ds.beached.values == 1)
    
    if any == True:
        l_box = np.any(np.logical_and(l_lon, l_lat), axis=1)
    else:
        l_box = np.logical_and(l_lon, l_lat)
    
    return l_box

def _get_particle_release_time_box(ds:xr.Dataset, l_box:np.ndarray[bool]) -> np.ndarray[datetime]:
    _, t_release = _get_particle_release_time_and_pid_indices(ds)
    time_release = ds.time.values[t_release[l_box]]
    return time_release

def _get_particle_entry_time_box(ds:xr.Dataset, box:dict, beached=False) -> np.ndarray[datetime]:
    
    l_box = get_l_particles_in_box(ds, box, any=False, beached=False)
    p_box, t_box = np.where(l_box)
    _, i_sort = np.unique(p_box, return_index=True)
    t_entry = t_box[i_sort]
    
    time_entry = ds.time[t_entry]
    return np.array(time_entry)

def get_n_particles_per_month_release_arrival(ds:xr.Dataset, l_box:np.ndarray[bool], island:dict,
                                              beached=False):
    release_time = _get_particle_release_time_box(ds, l_box)
    release_months = np.array([x.month for x in pd.to_datetime(release_time)])
    entry_time = _get_particle_entry_time_box(ds, island, beached=beached)
    entry_months = np.array([x.month for x in pd.to_datetime(entry_time)])
    months = np.arange(1,13,1)
    n_release = []
    n_entry = []
    
    for month in months:
        n_release.append(np.sum(release_months==month))
        n_entry.append(np.sum(entry_months==month))
    
    n_entry_per_release_month = np.zeros((len(months), len(months)))
    for i in range(len(entry_months)):
        n_entry_per_release_month[entry_months[i]-1, release_months[i]-1] += 1
    
    return np.array(n_release), np.array(n_entry), n_entry_per_release_month

def _get_particle_release_locations_box(ds:xr.Dataset, l_box:np.ndarray[bool]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    lon0, lat0 = get_initial_particle_lon_lat(ds)
    return (lon0[l_box], lat0[l_box])

def _get_main_sources_lon_lat_n_particles(ds:xr.Dataset, l_box:np.ndarray[bool]) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
    lon0, lat0 = _get_particle_release_locations_box(ds, l_box)
    coordinates0 = []
    for i in range(len(lon0)):
        coordinates0.append([lon0[i], lat0[i]])
    coordinates0 = np.array(coordinates0)
    coordinates0_unique, counts_unique = np.unique(coordinates0, axis=0, return_counts=True)
    lon0_unique = coordinates0_unique[:, 0]
    lat0_unique = coordinates0_unique[:, 1]
    return (lon0_unique, lat0_unique, counts_unique)

def _get_original_source_based_on_lon0_lat0(lon0:np.ndarray[float], lat0:np.ndarray[float],
                                           n_closest=6) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    iot_sources = get_iot_sources(original=True)
    i_closest = get_index_closest_point(iot_sources.lon, iot_sources.lat, lon0, lat0, n_closest=n_closest)
    lon = iot_sources.lon[i_closest]
    lat = iot_sources.lat[i_closest]
    yearly_waste = np.sum(iot_sources.waste[i_closest], axis=1)
    return lon, lat, yearly_waste

def get_main_sources_at_island(ds:xr.Dataset, island:dict, beached=False) -> tuple[np.ndarray, np.ndarray]:
    l_box = get_l_particles_in_box(ds, island, beached=beached)
    if np.any(l_box):
        lon0, lat0, p0 = _get_main_sources_lon_lat_n_particles(ds, l_box)
    else:
        lon0 = None
        lat0 = None
        p0 = None
    
    return lon0, lat0, p0

def _write_possible_original_sources_to_file(lon0:np.ndarray[float], lat0:np.ndarray[float], waste0:np.ndarray[float], output_path:str):
    with open(output_path, 'w') as f:
        f.write(f'Locations of sources:\n')
        for i in range(len(lon0)):
            f.write(f'\nSource {i+1}: {lon0[i]}, {lat0[i]}, {waste0[i]}\n')
            lon0_org, lat0_org, waste0_org = _get_original_source_based_on_lon0_lat0(lon0[i], lat0[i])
            f.write('Original source locations:\n')
            for j in range(len(lon0_org)):
                f.write(f'{lon0_org[j]}, {lat0_org[j]}, {waste0_org[j]}\n')
    log.info(f'Wrote possible original source locations to: {output_path}')

def get_sorted_percentage_big_sources(lon0:np.ndarray[float],
                                      lat0:np.ndarray[float],
                                      waste0:np.ndarray[float],
                                      cutoff_small=10,
                                      cutoff_big=50,
                                      output_path=None) -> tuple[np.ndarray[int], np.ndarray[float]]:
    if lon0 is None:
        log.info('No sources found!')
        return None
    
    i_sort = np.argsort(waste0)[::-1]
    i_big = np.where(waste0[i_sort] >= cutoff_small)
    waste0_big = waste0[i_sort][i_big]
    lon0_big = lon0[i_sort][i_big]
    lat0_big = lat0[i_sort][i_big]
    i_biggest = waste0_big >= cutoff_big
    lon0_biggest = lon0_big[i_biggest]
    lat0_biggest = lat0_big[i_biggest]
    waste0_biggest = waste0_big[i_biggest]
    percentage_waste0_big = waste0_big/np.sum(waste0)*100
    
    if output_path is not None:
        _write_possible_original_sources_to_file(lon0_biggest, lat0_biggest, waste0_biggest, output_path)
    
    return i_sort[i_big], percentage_waste0_big
    
if __name__ == '__main__':
    b = [10, None, None, None, 10, 1, 1, 100, 100]
    r = [70, None, None, None, 270, 70, 270, 70, 270]
    forcing = ['hycom_ww3', 'hycom', 'hycom_ww3', 'hycom_cfsr', 'hycom_ww3', 'hycom_ww3',
               'hycom_ww3', 'hycom_ww3', 'hycom_ww3']
    
    write_sources = False
    
    for i in range(len(b)):
        if b[i] == None:
            description = forcing[i]
        else:
            description = f'{forcing[i]}_b{b[i]}_r{r[i]}'
        
        input_path = f'{get_dir_from_json("pts_output")}iot_{description}.nc'
        output_path = f'{get_dir_from_json("pts_processed")}iot_particles_{description}.nc'
        output_path_density = f'{get_dir_from_json("pts_processed")}iot_density_{description}.nc'
        
        # process particles
        if not os.path.exists(output_path):
            process_parcels_netcdf(input_path, output_path)
        else:
            log.info(f'Particles already processed, skipping: {output_path}')
        
        # write main sources
        if write_sources == True:
            ds = xr.load_dataset(output_path)
            
            cki, ci = get_island_boxes_from_toml()
            lon0_cki, lat0_cki, waste0_cki = get_main_sources_at_island(ds, cki)
            _, p_waste0_cki = get_sorted_percentage_big_sources(lon0_cki, lat0_cki, waste0_cki,
                                                                output_path=f'plots/processing/cki_sources_{description}.txt')
            lon0_ci, lat0_ci, waste0_ci = get_main_sources_at_island(ds, ci)
            _, p_waste0_ci = get_sorted_percentage_big_sources(lon0_ci, lat0_ci, waste0_ci,
                                                               output_path=f'plots/processing/ci_sources_{description}.txt')
            
        # process particle density
        if not os.path.exists(output_path_density):
            process_particle_density(output_path, output_path_density)
        else:
            log.info(f'Density already processed, skipping: {output_path_density}')