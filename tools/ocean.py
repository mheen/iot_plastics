import numpy as np
import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tools import log

class Grid:
    def __init__(self,dx,lon_range,lat_range,dy=None,periodic=False):
        self.dx = dx
        if not dy:
            self.dy = dx
        else:
            self.dy = dy
        self.lon_min = lon_range[0]
        self.lon_max = lon_range[1]
        self.lat_min = lat_range[0]
        self.lat_max = lat_range[1]
        self.lon = np.arange(self.lon_min,self.lon_max+self.dx,self.dx)
        self.lat = np.arange(self.lat_min,self.lat_max+self.dy,self.dy)
        self.lon_size = len(self.lon)
        self.lat_size = len(self.lat)
        self.periodic = periodic

    def get_index(self,lon,lat):
        lon = np.array(lon)
        lat = np.array(lat)
        # get lon index
        lon_index = np.floor((lon-self.lon_min)*1/self.dx)
        lon_index = np.array(lon_index)
        l_index_lon_over = lon_index >= abs(self.lon_max-self.lon_min)*1/self.dx
        if self.periodic:
            lon_index[l_index_lon_over] = 0
        else:
            lon_index[l_index_lon_over] = np.nan
        l_index_lon_under = lon_index < 0
        if self.periodic:
            lon_index[l_index_lon_under]
        else:
            lon_index[l_index_lon_under] = np.nan
        # get lat index
        lat_index = np.floor((lat-self.lat_min)*1/self.dy)
        lat_index = np.array(lat_index)
        l_index_lat_over = lat_index >= abs(self.lat_max-self.lat_min)*1/self.dy
        lat_index[l_index_lat_over] = np.nan        
        l_index_lat_under = lat_index<0
        lat_index[l_index_lat_under] = np.nan        
        return (lon_index,lat_index)

    @staticmethod
    def get_from_lon_lat_array(lon,lat):
        dx = np.round(np.unique(np.diff(lon))[0],2)
        dy = np.round(np.unique(np.diff(lat))[0],2)
        lon_range = [np.nanmin(lon),np.nanmax(lon)]
        lat_range = [np.nanmin(lat),np.nanmax(lat)]        
        log.warning(None,f'dx ({np.unique(np.diff(lon))[0]}) to create Grid rounded to 2 decimals: dx = {dx}')
        if dy != dx:
            log.warning(None,f'dy ({np.unique(np.diff(lat))[0]}) to create Grid rounded to 2 decimals: dy = {dy}')
        return Grid(dx,lon_range,lat_range,dy=dy)

class OceanBasins:
    def __init__(self):
        self.basin = []

    def determine_if_point_in_basin(self,basin_name,p_lon,p_lat):
        p_lon = np.array(p_lon)
        p_lat = np.array(p_lat)
        if basin_name.startswith('po'):
            basin_name = [basin_name[:2]+'_l'+basin_name[2:],basin_name[:2]+'_r'+basin_name[2:]]
        else:
            basin_name = [basin_name]
        l_in_basin = np.zeros(len(p_lon)).astype('bool')
        for i in range(len(basin_name)):
            basin = self.get_basin_polygon(basin_name[i])
            for p in range(len(p_lon)):
                point = Point(p_lon[p],p_lat[p])
                l_in_polygon = basin.polygon.contains(point)
                l_in_basin[p] = l_in_polygon or l_in_basin[p]
        return l_in_basin

    def get_basin_polygon(self,basin_name):
        for basin in self.basin:
            if basin.name == basin_name:
                return basin
        raise ValueError('Unknown ocean basin requested. Valid options are: "io","ao","po", and any of these with "_nh" or "_sh" added.')

    @staticmethod
    def read_from_shapefile(input_path='input/oceanbasins_polygons.shp'):
        ocean_basins = OceanBasins()
        sf = shapefile.Reader(input_path)        
        shape_records = sf.shapeRecords() # reads both shapes and records(->fields)
        for i in range(len(shape_records)):
            name = shape_records[i].record[1]
            points = shape_records[i].shape.points
            polygon = Polygon(points)
            ocean_basins.basin.append(OceanBasin(name,polygon))
        sf.close()
        return ocean_basins

class OceanBasin:
    def __init__(self,name,polygon):
        self.name = name
        self.polygon = polygon

class OceanBasinGrid:
    def __init__(self,basin_name,dx,lon_range=None,lat_range=None):        
        self.basin_name = basin_name
        if lon_range is None:
            lon_range = [-180,180]
        if lat_range is None:
            lat_range = [-90,90]
        self.grid = Grid(dx,lon_range,lat_range)
        lon,lat = np.meshgrid(self.grid.lon,self.grid.lat)
        self.in_basin = np.ones(lon.shape).astype('bool')
        ocean_basins = OceanBasins.read_from_shapefile()
        for i in range(lon.shape[0]):            
            self.in_basin[i,:] = ocean_basins.determine_if_point_in_basin(basin_name,lon[i,:],lat[i,:])