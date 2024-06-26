import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps
import numpy as np
from glob import glob

lon_range_default = [110.0, 154.0]
lat_range_default = [-46.0, -24.0]

def plot_contours(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                  lon_range=None, lat_range=None, ax=None, show=False,
                  fontsize=10, color='k', linewidths=1,
                  clevels=[50, 100, 200, 400, 800, 1000],
                  clabel=True) -> plt.axes:

    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range)
        
    def _fmt(x):
        s = f'{x:.0f}'
        return s

    # get masked arrays so contour labels fall within plot if range is limited
    mask = np.ones(h.shape).astype(bool)
    mask = mask & (lon>=lon_range[0]) & (lon<=lon_range[1])
    mask = mask & (lat>=lat_range[0]) & (lat<=lat_range[1])
    xm = np.ma.masked_where(~mask, lon)
    ym = np.ma.masked_where(~mask, lat)
    zm = np.ma.masked_where(~mask, h)

    cs = ax.contour(xm, ym, zm, levels=clevels,
                    colors=color, linewidths=linewidths, transform=ccrs.PlateCarree())
    if clabel is True:
        ax.clabel(cs, cs.levels, fontsize=fontsize, inline=True, fmt=_fmt)

    if show is True:
        plt.show()
    else:
        return ax

def plot_bathymetry(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                    lon_range=None, lat_range=None, ax=None, show=False,
                    cmap='BrBG', vmin=None, vmax=None) -> plt.axes:

    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range)
    
    c = ax.pcolormesh(lon, lat, h, cmap=cmap, vmin=vmin, vmax=vmax)

    l,b,w,h = ax.get_position().bounds
    fig = plt.gcf()
    cbax = fig.add_axes([l+w+0.02, b, 0.05*w, h])
    cbar = plt.colorbar(c, cax=cbax)
    cbar.set_label('Bathymetry (m)')

    if show is True:
        plt.show()
    else:
        return ax, c, cbar

def load_ne_bathymetry(input_dir=f'{get_dir_from_json("ne")}'):
    # Read shapefiles, sorted by depth
    shp_dict = {}
    files = glob(f'{input_dir}ne_10m_bathymetry_all/*.shp')
    assert len(files) > 0
    files.sort()
    depths = []
    for f in files:
        depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
        depths.append(depth)
        bbox = (90, -15, 160, 60)  # (x0, y0, x1, y1)
        nei = shapereader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei
    depths = np.array(depths)[::-1]  # sort from surface to bottom
    return depths, shp_dict

def plot_ne_bathymetry(ax:plt.axes, lon_range=None, lat_range=None):
    # Load data (14.8 MB file)
    depths_str, shp_dict = load_ne_bathymetry()

    # Construct a discrete colormap with colors corresponding to each depth
    depths = depths_str.astype(int)
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = BoundaryNorm(boundaries, N)
    blues_cm = colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))

    # Iterate and plot feature for each depth level
    for i, depth_str in enumerate(depths_str):
        ax.add_geometries(shp_dict[depth_str].geometries(),
                          crs=ccrs.PlateCarree(),
                          color=colors_depths[i])

    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default
    
    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]],
                   ccrs.PlateCarree())

    # Convert vector bathymetries to raster (saves a lot of disk space)
    # while leaving labels as vectors
    ax.set_rasterized(True)

def add_grid(ax:plt.axes, meridians:list, parallels:list,
              xmarkers:str, ymarkers:str, draw_grid:bool) -> plt.axes:

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_xticks(meridians, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_yticks(parallels, crs=ccrs.PlateCarree())

    if xmarkers == 'top':
        ax.xaxis.tick_top()
    if xmarkers == 'off':
        ax.set_xticklabels([])
    if ymarkers == 'right':
        ax.yaxis.tick_right()
    if ymarkers == 'off':
        ax.set_yticklabels([])

    if draw_grid is True:
        ax.grid(b=True, linewidth=0.5, color='k', linestyle=':', zorder=10)

    return ax

def plot_basic_map(ax:plt.axes, lon_range=None, lat_range=None,
                   meridians=None, parallels=None,
                   xmarkers='bottom', ymarkers='left',
                   draw_grid=False, full_resolution=False,
                   show_reefs=False) -> plt.axes:
    
    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default
    if meridians is None:
        meridians = np.arange(110.0, 160.0, 10.0)
    if parallels is None:
        parallels = np.arange(-46.0, -24.0, 4.0)
        
    if full_resolution == True:
        coast = cfeature.GSHHSFeature(scale="full")
        ax.add_feature(coast, linewidth=1, edgecolor='k', facecolor='#d2d2d2', zorder=2)
    else:
        ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='#d2d2d2', zorder=2)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
    
    if show_reefs == True:
        reefs = cfeature.NaturalEarthFeature(
            category='physical',
            name='reefs',
            scale='10m',
            facecolor='#d2d2d2')
        
        ax.add_feature(reefs, edgecolor='k', facecolor='#d2d2d2', zorder=2)
    
    ax = add_grid(ax, meridians, parallels, xmarkers, ymarkers, draw_grid)

    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]],
                   ccrs.PlateCarree())
    
    return ax
