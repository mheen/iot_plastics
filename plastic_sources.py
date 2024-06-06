from tools.land import LandMask
from tools.ocean import OceanBasins
from tools import log
import shapefile
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cftr

class RiverSources:
    def __init__(self,lon,lat,time,waste,waste_min,waste_max):
        self.lon = lon
        self.lat = lat
        self.time = time
        self.waste = waste
        self.waste_min = waste_min
        self.waste_max = waste_max

    def remove_low_sources(self, cutoff=10):
        total_waste = np.sum(np.round(self.waste), axis=1)
        i_sources = np.where(total_waste > cutoff)
        
        self.lon = self.lon[i_sources]
        self.lat = self.lat[i_sources]
        self.waste = self.waste[i_sources]
        self.waste_min = self.waste_min[i_sources]
        self.waste_max = self.waste_max[i_sources]

    def convert_for_parcels_per_time(self, t, time, cutoff=10):
        self.remove_low_sources(cutoff=cutoff)
        lon0 = []
        lat0 = []
        time0 = []
        source_points = np.where(self.waste[:, t] != 0)[0]
        for p in source_points:
            n_particles = self.waste[p, t]
            lon0 = np.append(lon0, np.repeat(self.lon[p], n_particles))
            lat0 = np.append(lat0, np.repeat(self.lat[p], n_particles))
            time0 = np.append(time0, np.repeat(time, n_particles))
        return lon0, lat0, time0

    def get_riversources_from_ocean_basin(self,basin_name):
        ocean_basins = OceanBasins.read_from_shapefile()
        l_in_basin = ocean_basins.determine_if_point_in_basin(basin_name,self.lon,self.lat)
        lon = self.lon[l_in_basin]
        lat = self.lat[l_in_basin]
        waste = self.waste[l_in_basin,:]
        waste_min = self.waste_min[l_in_basin,:]
        waste_max = self.waste_max[l_in_basin,:]
        return RiverSources(lon,lat,self.time,waste,waste_min,waste_max)

    def get_riversources_in_lon_lat_range(self, lon_range, lat_range):
        l_lon = np.logical_and(lon_range[0]<=self.lon, self.lon<=lon_range[1])
        l_lat = np.logical_and(lat_range[0]<=self.lat, self.lat<=lat_range[1])
        l_range = np.logical_and(l_lon, l_lat)
        lon = self.lon[l_range]
        lat = self.lat[l_range]
        waste = self.waste[l_range, :]
        waste_min = self.waste_min[l_range, :]
        waste_max = self.waste_max[l_range, :]
        return RiverSources(lon, lat, self.time, waste, waste_min, waste_max)

    def move_sources_to_ocean(self,land_mask=LandMask.read_from_netcdf('input/hycom_landmask.nc'),
                              log_file='move_riversources_hycom_landmask.log'):
        lon = self.lon
        lat = self.lat        
        lm_halo = land_mask.get_landmask_with_halo()
        log.info(log_file,'Increased land mask with 1 grid cell.')
        log.info(log_file,'Getting indices of points on land...')
        p_land = self._get_index_of_sources_on_land(lm_halo)
        log.info(log_file,f'Found {str(len(p_land))} sources on land.')
        for i,p in enumerate(p_land):
            log.info(log_file,f'Trying to move source {str(i+1)}/{str(len(p_land))} , p = {str(p)}:')
            lon[p],lat[p] = lm_halo.get_closest_ocean_point(self.lon[p],self.lat[p],log_file=log_file)
        return RiverSources(lon,lat,self.time,self.waste,self.waste_min,self.waste_max)

    def _get_index_of_sources_on_land(self,lm):        
        p_land = []
        for p in range(len(self.lon)):
            mask = lm.get_mask_value(self.lon[p],self.lat[p])
            if mask == 1:
                p_land.append(p)
        return p_land

    def plot_org_and_moved(self,lon_range=[20.,130.],lat_range=[-40.,40.],
                           land_mask=LandMask.read_from_netcdf('input/hycom_landmask.nc'),
                           plot_mplstyle='plot_tools/plot.mplstyle'):
        original = self.read_from_shapefile()
        plt.style.use(plot_mplstyle)
        fig = plt.figure()
        ax = plt.gca(projection=ccrs.PlateCarree())
        ax.set_extent([-180,180,-80,80],ccrs.PlateCarree())
        ax.add_feature(cftr.COASTLINE,edgecolor='k',zorder=2)
        if land_mask is not None:
            # plot land mask            
            cmap_lm = colors.ListedColormap(['#ffffff','#000000'])
            norm_lm = colors.BoundaryNorm([0,0.5,1],cmap_lm.N,clip=True)
            l_lon = np.logical_and(land_mask.lon>=lon_range[0],land_mask.lon<=lon_range[-1])
            l_lat = np.logical_and(land_mask.lat>=lat_range[0],land_mask.lat<=lat_range[-1])
            lon_edges,lat_edges = land_mask.get_edges_from_center_points(l_lon=l_lon,l_lat=l_lat)
            ax.pcolormesh(lon_edges,lat_edges,land_mask.mask[l_lat,:][:,l_lon],
                          norm=norm_lm,cmap=cmap_lm,transform=ccrs.PlateCarree(),zorder=1)
        # plot source locations with nonzero waste input
        i_source = np.where(np.sum(self.waste,axis=1) != 0)[0]
        ax.scatter(original.lon[i_source],original.lat[i_source],marker='x',
                   c='#DC3015',label='original source locations',transform=ccrs.PlateCarree(),
                   zorder=3)
        ax.scatter(self.lon[i_source],self.lat[i_source],marker='o',c='#0C59A6',
                   label='source locations in ocean',transform=ccrs.PlateCarree(),zorder=4)        
        # legend
        patch_lm = Patch(facecolor='#000000',edgecolor='#000000',label='Hycom landmask')
        ax.legend()
        ax.set_title('River plastic source locations')
        plt.show()

    def plot(self,t=0,lon_range=None,lat_range=None,
             plot_mplstyle='plot_tools/plot.mplstyle',
             land_mask=LandMask.read_from_netcdf('input/hycom_landmask.nc')):
        plastic = np.round(self.waste[:,0])
        plastic[plastic==0] = np.nan
        plastic_log = np.log10(plastic)
        month = datetime.strptime(str(t+1),'%m').strftime('%b')
        plt.style.use(plot_mplstyle)
        fig = plt.figure()
        ax = plt.gca(projection=ccrs.PlateCarree())
        # map
        if not lon_range and not lat_range:
            lon_range = [-180,180]
            lat_range = [-80,80]
        ax.set_extent([lon_range[0],lon_range[1],lat_range[0],lat_range[1]],ccrs.PlateCarree())
        if land_mask is not None:
            # plot land mask
            cmap_lm = colors.ListedColormap(['#ffffff','#000000'])
            norm_lm = colors.BoundaryNorm([0,0.5,1],cmap_lm.N,clip=True)
            l_lon = np.logical_and(land_mask.lon>=lon_range[0],land_mask.lon<=lon_range[-1])
            l_lat = np.logical_and(land_mask.lat>=lat_range[0],land_mask.lat<=lat_range[-1])
            lon_edges,lat_edges = land_mask.get_edges_from_center_points(l_lon=l_lon,l_lat=l_lat)
            ax.pcolormesh(lon_edges,lat_edges,land_mask.mask[l_lat,:][:,l_lon],
                          norm=norm_lm,cmap=cmap_lm,transform=ccrs.PlateCarree(),zorder=1)
        else:
            ax.add_feature(cftr.COASTLINE,edgecolor='k',zorder=2)
        # set plot ranges and custom colormap
        ticks = np.array([1,10,10**2,10**3,10**4,10**5,10**6])
        ticks_str = ['1','10','10$^2$','10$^3$','10$^4$','10$^5$','10$^6$']
        levels = np.log10(ticks)
        cmap = colors.ListedColormap(['#63676b','#106bb8','#f1d435','#d50000','#640000','#470000'])
        norm = colors.BoundaryNorm(levels, cmap.N, clip=True)        
        c = ax.scatter(self.lon,self.lat,c=plastic_log,cmap=cmap,norm=norm,s=30,
                       transform=ccrs.PlateCarree(),zorder=3)
        # colorbar
        cbar = plt.colorbar(c,ticks=levels)
        cbar.ax.set_yticklabels(ticks_str)
        cbar.set_label('Plastic waste input [tonnes]')
        ax.set_title('River plastic input in '+month)
        plt.show()

    def plot_io(self,t=0):
        lon_range = [20.,130.]
        lat_range = [-56.,40.]        
        self.plot(t=t,lon_range=lon_range,lat_range=lat_range)

    def write_to_netcdf(self,output_path):
        nc = Dataset(output_path,'w',format='NETCDF4')
        # define dimensions
        nc.createDimension('points',len(self.lon))        
        nc.createDimension('time',len(self.time))
        # define variables
        nc_lon = nc.createVariable('lon',float,'points',zlib=True)
        nc_lat = nc.createVariable('lat',float,'points',zlib=True)
        nc_time = nc.createVariable('time',float,'time',zlib=True)
        nc_waste = nc.createVariable('waste',float,('points','time'),zlib=True)
        nc_waste_min = nc.createVariable('waste_min',float,('points','time'),zlib=True)
        nc_waste_max = nc.createVariable('waste_max',float,('points','time'),zlib=True)
        # write variables
        nc_lon[:] = self.lon
        nc_lat[:] = self.lat
        nc_time[:] = self.time
        nc_waste[:] = self.waste
        nc_waste_min[:] = self.waste_min
        nc_waste_max[:] = self.waste_max
        nc_waste.units = 'tonnes'
        nc.close()

    @staticmethod
    def read_from_netcdf(input_path='input/PlasticRiverSources_Lebreton2017_Hycom.nc'):        
        data = Dataset(input_path)
        lon = data['lon'][:]
        lat = data['lat'][:]
        time = data['time'][:]
        waste = data['waste'][:]
        waste_min = data['waste_min'][:]
        waste_max = data['waste_max'][:]
        return RiverSources(lon,lat,time,waste,waste_min,waste_max)

    @staticmethod
    def read_from_shapefile(input_path='input/PlasticRiverInputs_Lebreton2017.shp'):
        sf = shapefile.Reader(input_path)
        fields = sf.fields[1:] # ignore first field: DeletionFlag
        shape_records = sf.shapeRecords() # reads both shapes and records(->fields)
        lon = np.zeros((len(shape_records)))
        lat = np.zeros((len(shape_records)))
        waste = np.zeros((len(shape_records),12))
        waste_min = np.zeros((len(shape_records),12))
        waste_max = np.zeros((len(shape_records),12))
        time = np.arange(1,13,1) # waste input for every month
        for i in range(len(shape_records)):
            lon[i],lat[i] = shape_records[i].shape.points[0]
            for t in range(12): # monthly waste inputs
                month = datetime.strptime(str(t+1),'%m').strftime('%b').lower()
                t_mid = 3+3*t
                t_low = 4+3*t
                t_high = 5+3*t
                test_mid_str = 'i_mid_'+month
                test_low_str = 'i_low_'+month
                test_high_str = 'i_high_'+month
                if fields[t_mid][0] == test_mid_str:
                    if shape_records[i].record[t_mid] is not None:
                        waste[i,t] = shape_records[i].record[t_mid]
                else:
                    raise ValueError('i='+str(i)+',t='+str(t)+'Requested field does not match '+test_mid_str+', but is: '+fields[t_mid][0])
                if fields[t_low][0] == test_low_str:
                    if shape_records[i].record[t_low] is not None:
                        waste_min[i,t] = shape_records[i].record[t_low]
                else:
                    raise ValueError('i='+str(i)+',t='+str(t)+'Requested field does not match '+test_low_str+', but is: '+fields[t_low][0])
                if fields[t_high][0] == test_high_str:
                    if shape_records[i].record[t_high] is not None:
                        waste_max[i,t] = shape_records[i].record[t_high]
                else:
                    raise ValueError('i='+str(i)+',t='+str(t)+'Requested field does not match '+test_high_str+', but is: '+fields[t_high][0])
        return RiverSources(lon,lat,time,waste,waste_min,waste_max)
    
def get_iot_sources(cutoff=10, original=False) -> RiverSources:
    if original == False:
        global_sources = RiverSources.read_from_netcdf()
    else:
        global_sources = RiverSources.read_from_shapefile()
    iot_sources = global_sources.get_riversources_from_ocean_basin('iot')
    iot_sources.remove_low_sources(cutoff=cutoff)
    return iot_sources
