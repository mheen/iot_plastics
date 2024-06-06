from tools.ocean import Grid
from plot_tools.basic_maps import plot_basic_map
from tools.arrays import get_closest_index, get_matrix_value_or_nan
from tools.coordinates import get_distance_between_points
from tools import log
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
import numpy as np
import toml

def get_islands_from_toml(input_path='input/islands.toml'):
    full_config = toml.load(input_path)
    return full_config['cki_land'], full_config['ci_land']

def get_island_boxes_from_toml(input_path='input/islands.toml'):
    full_config = toml.load(input_path)
    return full_config['cki'], full_config['ci']

class LandMask:
    def __init__(self, lon, lat, mask):
        self.lon = lon
        self.lat = lat
        self.mask = mask # 0: ocean, 1: land

    def extend_iot_land(self, input_path='input/islands.toml'):
        cki, ci = get_islands_from_toml(input_path=input_path)
        
        i_lon_cki = get_closest_index(self.lon, cki['lon_range'])
        i_lat_cki = get_closest_index(self.lat, cki['lat_range'])
        i_lon_ci = get_closest_index(self.lon, ci['lon_range'])
        i_lat_ci = get_closest_index(self.lat, ci['lat_range'])
        
        self.mask[i_lat_cki[0]:i_lat_cki[1], i_lon_cki[0]:i_lon_cki[1]] = 1
        self.mask[i_lat_ci[0]:i_lat_ci[1], i_lon_ci[0]:i_lon_ci[1]] = 1

    def get_landmask_with_halo(self):
        '''Increases the size of the landmask by 1 gridcell.
        This can be used to move plastic
        source locations further away from land.'''
        i,j = np.where(self.mask==1)
        ip1 = np.copy(i)
        ip1[i<len(self.lat)-1] += 1 # i+1 but preventing final point increasing out of range
        jp1 = np.copy(j)
        jp1[j<len(self.lon)-1] += 1 # j+1 but preventing final point increasing out of range
        im1 = np.copy(i)
        im1[i>0] -= 1 # i-1 but preventing first point decreasing out of range
        jm1 = np.copy(j)
        jm1[j>0] -= 1 # j-1 but preventing first point decreasing out of range
        mask = np.copy(self.mask)        
        mask[ip1,j] = 1 # extend mask up
        mask[i,jp1] = 1 # extend mask right
        mask[ip1,jp1] = 1 # extend mask upper right
        mask[im1,j] = 1 # extend mask down
        mask[i,jm1] = 1 # extend mask left
        mask[im1,jm1] = 1 # extend mask lower left
        # (note: this was corrected after creating v3 of river sources)
        mask[ip1,jm1] = 1 # extend mask upper left
        mask[im1,jp1] = 1 # extend mask lower right
        return LandMask(self.lon,self.lat,mask)

    def get_mask_value(self,p_lon,p_lat):
        j,i = self.get_index(p_lon,p_lat)        
        return self.mask[i,j]

    def get_multiple_mask_values(self,p_lon,p_lat):
        j = get_closest_index(self.lon,p_lon)
        i = get_closest_index(self.lat,p_lat)
        return self.mask[i,j]

    def get_closest_ocean_point(self,p_lon,p_lat,log_file=None):
        j,i = self.get_index(p_lon,p_lat)        
        domain_boundaries = self._get_mininum_surrounding_domain_including_ocean(i,j,log_file)
        if domain_boundaries is not None:
            lon_ocean,lat_ocean = self._get_ocean_coordinates(domain_boundaries,log_file)
            distances = np.empty((len(lon_ocean)))*np.nan
            for p in range(len(lon_ocean)):
                distances[p] = get_distance_between_points(p_lon,p_lat,lon_ocean[p],lat_ocean[p])
            p_closest = np.where(distances==np.nanmin(distances))[0][0]
            lon_closest = lon_ocean[p_closest]
            lat_closest = lat_ocean[p_closest]
            if log_file is not None:
                log.info(log_file,'Found closest ocean point: '+str(lon_closest)+', '+str(lat_closest)+
                         ' to point: '+str(p_lon)+', '+str(p_lat)+'.')
            return lon_closest,lat_closest
        return np.nan,np.nan    

    def get_index(self,p_lon,p_lat):
        dlon = abs(self.lon-p_lon)
        dlat = abs(self.lat-p_lat)
        j = np.where(dlon==np.nanmin(dlon))[0][0]
        i = np.where(dlat==np.nanmin(dlat))[0][0]        
        return j,i

    def get_edges_from_center_points(self,l_lon=None,l_lat=None):
        if l_lon is None:
            l_lon = np.ones(len(self.lon)).astype('bool')
        if l_lat is None:
            l_lat = np.ones(len(self.lat)).astype('bool')
        # convert lon and lat from center points (e.g. HYCOM) to edges (pcolormesh)
        lon_center = self.lon[l_lon]
        dlon = np.diff(lon_center)
        for i in range(len(lon_center)):
            if i == 0:
                lon_edges = lon_center[i]-0.5*dlon[i]
                lon_pcolor = np.append(lon_edges,lon_center[i]+0.5*dlon[i])
            elif i == len(lon_center)-1:
                lon_edges = np.append(lon_edges,lon_center[i]+0.5*dlon[i-1])
            else:
                lon_edges= np.append(lon_edges,lon_center[i]+0.5*dlon[i])        
        lat_center = self.lat[l_lat]
        dlat = np.diff(lat_center)
        for i in range(len(lat_center)):
            if i == 0:
                lat_edges = lat_center[i]-0.5*dlat[i]
                lat_edges= np.append(lat_edges,lat_center[i]+0.5*dlat[i])
            elif i == len(lat_center)-1:
                lat_edges = np.append(lat_edges,lat_center[i]+0.5*dlat[i-1])
            else:
                lat_edges = np.append(lat_edges,lat_center[i]+0.5*dlat[i])
        return lon_edges,lat_edges

    def _get_ocean_coordinates(self,domain_boundaries,log_file):
        i_min = domain_boundaries[0]
        i_max = domain_boundaries[1]
        j_min = domain_boundaries[2]
        j_max = domain_boundaries[3]
        lon = self.lon[j_min:j_max]
        lat = self.lat[i_min:i_max]
        dlon = np.append(np.diff(lon),np.diff(lon)[-1])
        dlat = np.append(np.diff(lat),np.diff(lat)[-1])
        ocean = self.mask[i_min:i_max,j_min:j_max] == 0
        i_ocean,j_ocean = np.where(ocean)
        # lon and lat in center of grid points:
        lon_ocean = lon[j_ocean]+dlon[j_ocean]/2
        lat_ocean = lat[i_ocean]+dlat[j_ocean]/2
        if log_file is not None:
            log.info(log_file,'Found '+str(len(lon_ocean))+' ocean points.')
        return lon_ocean,lat_ocean

    def _get_mininum_surrounding_domain_including_ocean(self,i,j,log_file):
        '''Increases number of grid cells around a specific
        point until an ocean cell is included in the domain.'''
        for n in range(50):            
            n_cells = 10+n*10
            if log_file is not None:
                log.info(log_file,'Finding domain size with ocean: n_cells='+str(n_cells))
            i_min,i_max = self._get_min_max_indices(i,n_cells,'i')
            j_min,j_max = self._get_min_max_indices(j,n_cells,'j')
            land_mask = self.mask[i_min:i_max,j_min:j_max]
            ocean = land_mask == 0
            if ocean.any():
                if log_file is not None:
                    log.info(log_file,'Success.')
                domain_boundaries = [i_min,i_max,j_min,j_max]
                return domain_boundaries
        log.info(log_file,'Did not find a boundary within n_cells='+str(n_cells)+', skipping point.')
        return None

    def _get_min_max_indices(self,i,n,i_type):
        i_min = i-n
        i_max = i+n+1
        if i_type == 'i':
            len_i = self.mask.shape[0]
        elif i_type == 'j':
            len_i = self.mask.shape[1]
        else:
            raise ValueError('Unknown i_type to get indices, should be either "i" or "j".')
        if i_min >= 0 and i_max <= len_i:
            return (i_min,i_max)
        elif i_min < 0 and i_max <= len_i:
            return (0,i_max)
        elif i_max > len_i and i_min >= 0:
            return (i_min,len_i)
        elif i_min < 0 and i_max > len_i:
            return(0,len_i)
        else:
            raise ValueError('Error getting '+i_type+' indices: '+i_type+'='+str(i)+',n='+str(n)) 

    def plot(self, lon_range=None, lat_range=None):
        if lon_range == None:
            lon_range = [-180, 180]
        if lat_range == None:
            lat_range = [-80, 80]
        
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range, lat_range)
        ax.pcolormesh(self.lon, self.lat, self.mask, transform=ccrs.PlateCarree())
        plt.show()

    def write_to_netcdf(self, output_path):
        ds = xr.Dataset(data_vars=dict(
                            mask=(['lat', 'lon'], self.mask)),
                        coords=dict(
                            lon=self.lon,
                            lat=self.lat))
        
        ds.to_netcdf(path=output_path, mode='w')
        log.info(f'Wrote landmask to {output_path}')

    @staticmethod
    def read_from_netcdf(input_path='input/hycom_landmask.nc'):
        ds = xr.load_dataset(input_path)
        lon = ds.lon.values
        lat = ds.lat.values
        mask = ds.mask.values
        return LandMask(lon, lat, mask)

    @staticmethod
    def get_mask_from_vel(input_path):
        ds = xr.load_dataset(input_path)
        lon = ds.lon.values
        lat = ds.lat.values
        
        if len(ds['u'][:].shape) == 3:
            u = ds['u'][0,:,:].values
        else:
            u = ds['u'].values
        mask = np.isnan(u).astype('int')
        return LandMask(lon, lat, mask)
