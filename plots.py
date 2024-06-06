from plastic_sources import get_iot_sources
from processing import get_l_particles_in_box, get_main_sources_at_island, get_sorted_percentage_big_sources
from processing import get_n_particles_per_month_release_arrival
from tools.observations import read_iot_plastic_type_counts, get_monthly_plastic_samples
from tools.land import get_island_boxes_from_toml
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import plot_box, add_subtitle, color_y_axis
from processing import get_iot_lon_lat_range
from tools.files import get_dir_from_json
from tools import log
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import xarray as xr
import os

lon_range, lat_range = get_iot_lon_lat_range()
meridians = [90, 100, 110, 120]
parallels = [-20, -10, 0, 10, 20]

def get_colormap_reds(n):
    colors = ['#fece6b','#fd8e3c','#f84627','#d00d20','#b50026','#950026','#830026']
    return colors[:n]

def get_colormap_linestyles_black(n):
    colors = ['k', 'k', 'k', 'k', '#636363', '#636363', '#636363', '#636363']
    linestyles = ['-', '-.', ':', '--', '-', '-.', ':', '--']
    return colors[:n], linestyles[:n]

def get_months_colors():
    colors = ['#ffedbc', '#fece6b', '#fdc374', '#fb9d59', '#f57547', '#d00d20',
    '#c9e7f1', '#90c3dd', '#4576b4', '#000086', '#4d00aa', '#30006a']
    return colors

def _source_info() -> tuple[np.ndarray[float], list[str], list[str], list[float], list[float], list[float]]:
    ranges = np.array([350000., 20000., 5000., 1000., 100., 10., 0.])
    labels = ['> 20,000','5,000 - 20,000','1,000 - 5,000','100 - 1,000','10 - 100','< 10']
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    edge_widths = [0.7,0.7,0.7,0.5,0.5,0.5]
    sizes = [10,5,4,3,2,1]*6
    legend_sizes = [6,5,4,3,2,1]
    return (ranges, labels, colors, edge_widths, sizes, legend_sizes)

def _iot_source_info() -> tuple[np.ndarray[float], list[str], list[str], list[float], list[float], list[float]]:
    ranges = np.array([500., 200., 100., 50., 25., 10., 1.])
    labels = ['> 200','100 - 200','50 - 100','25 - 50', '10 - 25', '< 10']
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    edge_widths = [0.7, 0.7, 0.7, 0.5, 0.0, 0.0]
    sizes = [6, 5, 4, 3, 2, 1]
    legend_sizes = [6, 5, 4, 3, 2, 1]
    return (ranges, labels, colors, edge_widths, sizes, legend_sizes)

def _get_colors_for_samples(counts:np.ndarray[int], ranges:list[float]):
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    
    sample_colors = []
    for waste in counts:
        l_range = []
        for i in range(len(colors)):
            l_range.append(ranges[i] >= waste >= ranges[i+1])
        i_range = np.where(l_range)[0][0]
        sample_colors.append(colors[i_range])
    
    return sample_colors

def _get_marker_colors_sizes_edgewidths_for_sources(waste_input:np.ndarray[int],
                                                    sources_type='all') -> tuple[np.ndarray[str], np.ndarray[float], np.ndarray[float]]:
    if sources_type == 'all':
        (ranges, _, colors, edge_widths, sizes, _) = _source_info()
    elif sources_type == 'iot':
        (ranges, _, colors, edge_widths, sizes, _) = _iot_source_info()
    else:
        raise ValueError(f'Unknown sources type: {sources_type}. Valid options are "all" and "iot".')
    source_colors = []
    source_sizes = []    
    source_edge_widths = []
    for waste in waste_input:
        l_range = []
        for i in range(len(colors)):
            l_range.append(ranges[i] >= waste >= ranges[i+1])
        i_range = np.where(l_range)[0][0]
        source_colors.append(colors[i_range])
        source_sizes.append(sizes[i_range])
        source_edge_widths.append(edge_widths[i_range])
    return np.array(source_colors), np.array(source_sizes), np.array(source_edge_widths)

def _get_legend_entries_for_sources(sources_type='all') -> list:
    if sources_type == 'all':
        (_, labels, colors, edge_widths, _, legend_sizes) = _source_info()
    elif sources_type == 'iot':
        (_, labels, colors, edge_widths, _, legend_sizes) = _iot_source_info()
    else:
        raise ValueError(f'Unknown sources type: {sources_type}. Valid options are "all" and "iot".')
    legend_entries = []
    for i in range(len(colors)):
        legend_entries.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],                              
                              markersize=legend_sizes[i],label=labels[i],markeredgewidth=edge_widths[i]))
    return legend_entries

def _match_river_locations_to_names(lon0:np.ndarray[float], lat0:np.ndarray[float],
                                    input_path='input/rivers.csv') -> list[str]:
    df = pd.read_csv(input_path)
    rivers = []
    for i in range(len(lon0)):
        l_river = np.logical_and(np.round(df['model_lon'], 2) == np.round(lon0[i], 2),
                                 np.round(df['model_lat'], 2) == np.round(lat0[i], 2))
        if not np.any(l_river):
            rivers.append('')
            continue
        rivers.append(df['name'].values[l_river][0])
        
    return rivers

def _match_river_names_to_locations(rivers:list[str],
                                    input_path='input/rivers.csv') -> tuple[np.ndarray[float], np.ndarray[float]]:
    df = pd.read_csv(input_path)
    lon = []
    lat = []
    for i in range(len(rivers)):
        l_river = df['name']==rivers[i]
        if np.any(l_river) == False:
            log.info(f'Did not find a river {rivers[i]}, skipping.')
            continue    
        lon.append(df['org_lon'].values[l_river][0])
        lat.append(df['org_lat'].values[l_river][0])  
        
    return np.array(lon), np.array(lat)

def _find_n_rivers_contributing_x_percent(p0_big:np.ndarray[float], x_percent:float) -> int:
    p = 0.0
    n = 0
    for n in range(1, len(p0_big)):
        p = np.sum(p0_big[:n])
        if p >= x_percent:
            return n
    raise ValueError(f'Not enough rivers to contribute {x_percent}.')
    
def figure1_overview(output_path=None,
                     show=True,
                     river_dir = get_dir_from_json('indonesia_rivers'),
                     rivers=[],
                     river_color='#002eb5',
                     input_path_cities='input/cities.csv',
                     cities=[],
                     city_color='#d00d20'):

    cki, ci = get_island_boxes_from_toml()
    lon_range_java = [102.0, 116.0]
    lat_range_java = [-9.5, -4.0]
    meridians_java = [105.0, 110.0, 115.0]
    parallels_java = [-8.5, -7.0, -5.5]

    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(hspace=0.05)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    # (a) Overview NE monsoon
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, lon_range, lat_range, meridians, parallels, xmarkers='off')
    plot_box(ax1, cki['lon_range'], cki['lat_range'])
    plot_box(ax1, ci['lon_range'], ci['lat_range'])
    add_subtitle(ax1, '(a) NE monsoon (DJF) currents')
    
    # (b) Overview SW monsoon
    ax3 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, lon_range, lat_range, meridians, parallels, xmarkers='off', ymarkers='off')
    plot_box(ax3, cki['lon_range'], cki['lat_range'])
    plot_box(ax3, ci['lon_range'], ci['lat_range'])
    add_subtitle(ax3, '(b) SW monsoon (JJA) currents')
    
    # (c) Indonesian river sources
    ax4 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax4 = plot_basic_map(ax4, lon_range, lat_range, meridians, parallels)
    plot_box(ax4, cki['lon_range'], cki['lat_range'])
    plot_box(ax4, ci['lon_range'], ci['lat_range'])
    
    iot_sources = get_iot_sources()
    iot_waste = np.sum(iot_sources.waste, axis=1)
    i_sort = np.argsort(iot_waste)

    (iot_colors,
    iot_sizes,
    iot_edge_widths) = _get_marker_colors_sizes_edgewidths_for_sources(iot_waste)
    
    ax4.scatter(iot_sources.lon[i_sort], iot_sources.lat[i_sort], marker='o', c=iot_colors[i_sort],
                s=np.array(iot_sizes[i_sort])*6, linewidths=iot_edge_widths[i_sort], edgecolors='k',
                zorder=3)
    add_subtitle(ax4, '(c) River plastic sources')
    legend_entries = _get_legend_entries_for_sources()
    ax4.set_anchor('E')
    l = ax4.legend(handles=legend_entries, title='(tonnes/year)', loc='upper left',
                   ncol=3, columnspacing=0.3)
    
    # (d) zoom Java
    ax2 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, lon_range_java, lat_range_java, meridians_java, parallels_java)
    add_subtitle(ax2, '(d) Main rivers IOT waste')
    # plot rivers
    for i in range(len(rivers)):
        reader = shpreader.Reader(f'{river_dir}{rivers[i]}.shp')
        river_lines = reader.records()
        for river_line in river_lines:
            ax2.add_geometries([river_line.geometry], ccrs.PlateCarree(), edgecolor=river_color,
                               facecolor='None', zorder=5, linewidth=1.5)
    # plot cities
    df = pd.read_csv(input_path_cities)
    for i in range(len(cities)):
        l_city = df['name']==cities[i]
        if np.any(l_city):
            ax2.scatter(df['lon'].values[l_city], df['lat'].values[l_city], marker='o',
                        c=city_color, s=8, edgecolors='k', zorder=6)
    
    # move legend and ax4
    l.set_bbox_to_anchor((-0.2, -0.2))
    l1, b1, w1, h1 = ax1.get_position().bounds
    l4, b4, w4, h4 = ax4.get_position().bounds
    ax4.set_position([l1, b4, w4, h4])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def figure2_samples(output_path=None, show=True):
    # plastic samples
    (months, n_plastic_cki, n_plastic_ci, n_months_cki, n_months_ci) = get_monthly_plastic_samples()
    
    material_counts, category_counts, item_counts = read_iot_plastic_type_counts()
    
    # removing very low counts for clarity in plot
    material_counts = material_counts.drop(['Dangerous', 'Large', 'Admin'])
    category_counts = category_counts.drop('Miscellaneous Categories')
    
    # getting shorter names and bar colors
    material_name_conversion = pd.read_csv('input/material_name_conversion.csv')
    str_items_m = []
    for i in material_counts.index:
        str_items_m.append(material_name_conversion[material_name_conversion['org'] == i]['new'].values[0])
    colors_m = _get_colors_for_samples(material_counts.values, [2500, 500, 200, 0])
    
    category_name_conversion = pd.read_csv('input/category_name_conversion.csv')
    str_items_c = []
    for i in category_counts.index:
        str_items_c.append(category_name_conversion[category_name_conversion['org'] == i]['new'].values[0])
    colors_c = _get_colors_for_samples(category_counts.values, [900, 500, 325, 220, 160])
    
    item_name_conversion = pd.read_csv('input/item_name_conversion.csv')
    l_low = item_counts.values/np.sum(item_counts.values)*100 < 2.0
    str_items = []
    for i, item in enumerate(item_counts.index):
        if l_low[i] == True:
            str_items.append('')
            continue
        str_items.append(item_name_conversion[item_name_conversion['org'] == item]['new'].values[0])
    colors_i = _get_colors_for_samples(item_counts.values, [300, 100, 50, 10, 0])
    
    fig = plt.figure(figsize=(9, 10))
    plt.subplots_adjust(hspace=0.6)
    
    # (a) Materials collected
    ax1 = plt.subplot(3, 2, 1)
    ax1.bar(str_items_m, material_counts.values, zorder=5, color=colors_m, edgecolor='k')
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_ylim([0, 2500])
    ax1.set_yticks(np.arange(0, 2500, 500))
    ax1.grid(True, axis='y')
    ax1.tick_params('x', length=0)
    add_subtitle(ax1, '(a) Debris materials')
    
    # (b) Plastic types collected
    ax2 = plt.subplot(3, 2, 2)
    ax2.bar(str_items_c, category_counts.values, zorder=5, color=colors_c, edgecolor='k')
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.set_ylim([0, 1000])
    ax2.set_yticks(np.arange(0, 1000, 200))
    ax2.grid(True, axis='y')
    ax2.tick_params('x', length=0)
    add_subtitle(ax2, '(b) Plastic categories')
    
    # (c) Plastic items collected
    ax3 = plt.subplot(3, 2, (3, 4))
    ax3.bar(np.arange(0, len(item_counts)), item_counts.values, zorder=5, color=colors_i, edgecolor='k')
    ax3.set_xticks(np.arange(0, len(item_counts)))
    ax3.set_xticklabels(str_items)
    ax3.tick_params(axis='x', labelrotation=90)
    ax3.set_xlim([-1, len(item_counts)])
    ax3.set_ylim([0, 350])
    ax3.set_yticks(np.arange(0, 350, 50))
    ax3.grid(True, axis='y')
    ax3.tick_params('x', length=0)
    add_subtitle(ax3, '(c) Plastic items')
    
    # (d) Monthly # items CI
    ax4 = plt.subplot(3, 2, 5)
    # items collected
    ax4.bar(months-0.2, n_plastic_ci, width=0.4, color='#f57547', edgecolor='k', zorder=5)
    ax4.set_xticks(months)
    ax4.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax4.set_ylim([0, 140000])
    ax4.set_yticks(np.arange(0, 160000, 20000))
    ax4.grid(True, axis='y')
    ax4.tick_params('x', length=0)
    ax4.set_ylabel('Plastic items (#/month)')
    # beach clean-ups
    ax44 = ax4.twinx()
    ax44.bar(months+0.2, n_months_ci, width=0.4, color='#adadad', edgecolor='k', zorder=6)
    ax44.set_ylim([0, 14])
    ax44.set_yticks(np.arange(0, 16, 2))
    ax44.set_yticklabels([])
    color_y_axis(ax44, '#adadad', 'right')
    
    # legend
    legend_elements = [Patch(facecolor='#f57547', edgecolor='k', label='Plastic items'),
                       Patch(facecolor='#adadad', edgecolor='k', label='Beach clean-ups')]
    ax4.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.45, 0.9))
    
    add_subtitle(ax4, '(d) Plastic collected CI')

    # (e) Monthly # items CKI
    ax5 = plt.subplot(3, 2, 6)
    # items collected
    ax5.bar(months-0.2, n_plastic_cki, width=0.4, color='#f57547', edgecolor='k', zorder=5) # items collected
    ax5.set_xticks(months)
    ax5.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax5.set_ylim([0, 140000])
    ax5.set_yticks(np.arange(0, 160000, 20000))
    ax5.grid(True, axis='y')
    ax5.tick_params('x', length=0)
    ax5.set_yticklabels([])
    # beach clean-ups
    ax55 = ax5.twinx()
    ax55.bar(months+0.2, n_months_cki, width=0.4, color='#adadad', edgecolor='k', zorder=6)
    ax55.set_ylim([0, 14])
    ax55.set_yticks(np.arange(0, 16, 2))
    ax55.set_ylabel('Beach clean-ups (##/month)')
    color_y_axis(ax44, '#adadad', 'right')
    
    add_subtitle(ax5, '(e) Plastic collected CKI')
    
    # move ax3 up
    l3, b3, w3, h3 = ax3.get_position().bounds
    ax3.set_position([l3, b3+0.025, w3, h3])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    
def figure3_sources(ds_particles:xr.Dataset, cutoff_histogram=10,
                    output_path=None, show=True, title=''):
    
    cki, ci = get_island_boxes_from_toml()
    l_box_cki = get_l_particles_in_box(ds_particles, cki)
    l_box_ci = get_l_particles_in_box(ds_particles, ci)
    lon0_cki, lat0_cki, waste0_cki = get_main_sources_at_island(ds_particles, cki)
    lon0_ci, lat0_ci, waste0_ci = get_main_sources_at_island(ds_particles, ci)
    
    (colors0_cki,
    sizes0_cki,
    edgewidths0_cki) = _get_marker_colors_sizes_edgewidths_for_sources(waste0_cki, sources_type='iot')
    i_sort_cki = np.argsort(waste0_cki)
    (colors0_ci,
    sizes0_ci,
    edgewidths0_ci) = _get_marker_colors_sizes_edgewidths_for_sources(waste0_ci, sources_type='iot')
    i_sort_ci = np.argsort(waste0_ci)
    
    i0_big_cki, p0_big_cki = get_sorted_percentage_big_sources(lon0_cki, lat0_cki, waste0_cki, cutoff_small=cutoff_histogram)
    i0_big_ci, p0_big_ci = get_sorted_percentage_big_sources(lon0_ci, lat0_ci, waste0_ci, cutoff_small=cutoff_histogram)
    
    n50_cki = _find_n_rivers_contributing_x_percent(p0_big_cki, 50.0)
    n50_ci = _find_n_rivers_contributing_x_percent(p0_big_ci, 50.0)

    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    tracks_color = '#2e4999'
    
    # (a) sources CI map
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, lon_range, lat_range, meridians, parallels, xmarkers='off')
    plot_box(ax1, ci['lon_range'], ci['lat_range'])
    i_box_ci = np.where(l_box_ci)[0]
    for i in i_box_ci:
        ax1.plot(ds_particles.lon.values[i, :], ds_particles.lat.values[i, :], color=tracks_color, linewidth=0.1, zorder=0)
    ax1.scatter(lon0_ci[i_sort_ci], lat0_ci[i_sort_ci], marker='o',
                c=colors0_ci[i_sort_ci], s=np.array(sizes0_ci[i_sort_ci])*6, linewidths=edgewidths0_ci[i_sort_ci],
                edgecolors='k', zorder=4)
    add_subtitle(ax1, '(a) CI sources and tracks')
    
    # (b) CI histogram
    rivers_ci_50 = _match_river_locations_to_names(lon0_ci[i0_big_ci[:n50_ci]], lat0_ci[i0_big_ci[:n50_ci]])
    log.info(f'Rivers {rivers_ci_50} contribute {np.round(np.sum(p0_big_ci[:n50_ci]), 0)}% to waste on CI.')
    rivers_ci = _match_river_locations_to_names(lon0_ci[i0_big_ci], lat0_ci[i0_big_ci])
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(np.arange(0, len(p0_big_ci)), p0_big_ci, color=colors0_ci[i0_big_ci], zorder=5)
    ax2.set_ylim([0, 55])
    ax2.set_yticks(np.arange(0, 55, 5))
    ax2.set_ylabel('% particles arriving')
    
    ax2.set_xticks(np.arange(0, len(rivers_ci)))
    ax2.set_xticklabels(rivers_ci, rotation='vertical')
    ax2.grid(True, axis='y')
    ax2.tick_params('x', length=0)
    
    add_subtitle(ax2, '(b) River contributions to CI')
    
    # (c) sources CKI map
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, lon_range, lat_range, meridians, parallels)
    plot_box(ax3, cki['lon_range'], cki['lat_range'])
    i_box_cki = np.where(l_box_cki)[0]
    for i in i_box_cki:
        ax3.plot(ds_particles.lon.values[i, :], ds_particles.lat.values[i, :], color=tracks_color, linewidth=0.1, zorder=0)
    ax3.scatter(lon0_cki[i_sort_cki], lat0_cki[i_sort_cki], marker='o',
                c=colors0_cki[i_sort_cki], s=np.array(sizes0_cki[i_sort_cki])*6, linewidths=edgewidths0_cki[i_sort_cki],
                edgecolors='k', zorder=4)
    add_subtitle(ax3, '(c) CKI sources and tracks')
    
    # (d) CKI histogram
    rivers_cki_50 = _match_river_locations_to_names(lon0_cki[i0_big_cki[:n50_cki]], lat0_cki[i0_big_cki[:n50_cki]])
    log.info(f'Rivers {rivers_cki_50} contribute {np.round(np.sum(p0_big_cki[:n50_cki]), 0)}% to waste on CKI.')
    rivers_cki = _match_river_locations_to_names(lon0_cki[i0_big_cki], lat0_cki[i0_big_cki])

    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(np.arange(0, len(p0_big_cki)), p0_big_cki, color=colors0_cki[i0_big_cki], zorder=5)
    ax4.set_ylim([0, 55])
    ax4.set_yticks(np.arange(0, 55, 5))
    ax4.set_ylabel('% particles arriving')
    
    ax4.set_xticks(np.arange(0, len(rivers_cki)))
    ax4.set_xticklabels(rivers_cki, rotation='vertical')
    ax4.grid(True, axis='y')
    ax4.tick_params('x', length=0)
    
    add_subtitle(ax4, '(d) River contributions to CKI')

    # legend
    legend_entries = _get_legend_entries_for_sources(sources_type='iot')
    ax1.legend(handles=legend_entries, title='# particles', loc='upper left',
               bbox_to_anchor=(-0.25, -0.05), ncol=3, columnspacing=0.3)
    
    plt.suptitle(title, y=0.93)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        # write results to txt file for later reference if needed
        output_path_txt = f'{os.path.splitext(output_path)[0]}.txt'
        with open(output_path_txt, 'w') as f:
            f.write('CI:')
            for i in range(len(rivers_ci)):
                f.write(f'\n{rivers_ci[i]}, {p0_big_ci[i]}, {waste0_ci[i0_big_ci[i]]}')
            f.write('\n')
            f.write('\nCKI:')
            for i in range(len(rivers_cki)):
                f.write(f'\n{rivers_cki[i]}, {p0_big_cki[i]}, {waste0_cki[i0_big_cki[i]]}')
    
    if show == True:
        plt.show()
    else:
        plt.close()

def figure4_seasonality(ds_particles:xr.Dataset,
                        rivers=[],
                        output_path=None, show=True):
    
    cki, ci = get_island_boxes_from_toml()
    l_box_cki = get_l_particles_in_box(ds_particles, cki)
    l_box_ci = get_l_particles_in_box(ds_particles, ci)
    
    cki_n_release, _, cki_n_entry = get_n_particles_per_month_release_arrival(ds_particles, l_box_cki, cki)
    ci_n_release, _, ci_n_entry = get_n_particles_per_month_release_arrival(ds_particles, l_box_ci, ci)
    
    iot_sources = get_iot_sources(original=True)
    lon_rivers, lat_rivers = _match_river_names_to_locations(rivers)
    river_waste = []
    for i in range(len(rivers)):
        i_river = np.where(np.logical_and(iot_sources.lon==lon_rivers[i], iot_sources.lat==lat_rivers[i]))
        river_waste.append(np.squeeze(iot_sources.waste[i_river]))

    max_river_input = np.ceil(np.nanmax(river_waste))+np.ceil(0.3*np.nanmax(river_waste))

    month_colors = get_months_colors()
    river_colors, river_linestyles = get_colormap_linestyles_black(len(rivers))

    def _histogram_release_arrival(ax, n_release, n_entry, ylim=[0, 320]):
        p_release = n_release#/np.sum(n_release)*100
        p_entry = n_entry#/np.sum(n_entry)*100
        months = np.arange(1,13,1)
        colors = get_months_colors()
        ax.bar(months-0.2, p_release, width=0.4, label='Release', color=colors,
            hatch='////', edgecolor='k', zorder=5)
        n_entry_cumulative = p_entry.cumsum(axis=1)
        ax.bar(months+0.2, p_entry[:, 0], width=0.4, color=colors[0],
               edgecolor='k', zorder=5)
        for i in range(1, 11):
            heights = n_entry_cumulative[:, i]        
            starts = n_entry_cumulative[:, i-1]
            ax.bar(months+0.2, p_entry[:, i], bottom=n_entry_cumulative[:, i-1],
                    width=0.4, color=colors[i], edgecolor='k', zorder=5)
        ax.set_xticks(months)
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.set_ylabel('Particles (#/month)')
        ax.set_ylim(ylim)

    fig = plt.figure(figsize=(8, 7))

    # (a) seasonal waste input
    ax1 = plt.subplot(3, 2, (1, 2))
    for i in range(len(rivers)):
        ax1.plot(iot_sources.time, river_waste[i], label=rivers[i], color=river_colors[i], linestyle=river_linestyles[i])
    ax1.set_xticks(iot_sources.time)
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_xlim(1, 12)
    ax1.set_ylim(0, max_river_input)
    ax1.set_ylabel('Released particles\n(#/month)')
    ax1.legend(loc='upper right')
    
    add_subtitle(ax1, '(a) Seasonal input of plastic waste from main rivers')

    # (b) seasonal particles arriving CI
    ax2 = plt.subplot(3, 2, 3)
    _histogram_release_arrival(ax2, ci_n_release, ci_n_entry)
    add_subtitle(ax2, '(b) Seasonality reaching CI')
    
    # legend
    legend_elements = [Patch(facecolor=month_colors[3], edgecolor='k', hatch='//////', label='Release'),
                       Patch(facecolor=month_colors[3], edgecolor='k', label='Arrival')]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.5, 0.8))

    # (c) seasonal particles arriving CKI
    ax3 = plt.subplot(3, 2, 4)
    _histogram_release_arrival(ax3, cki_n_release, cki_n_entry)
    ax3.set_yticklabels([])
    ax3.set_ylabel('')
    add_subtitle(ax3, '(c) Seasonality reaching CKI')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    plot_f1 = False
    plot_f2 = False
    plot_f3 = True
    plot_f4 = True
    plot_f5 = False
    
    b = [None, None, None, 10, 10]
    r = [None, None, None, 70, 270]
    forcing = ['hycom', 'hycom_ww3', 'hycom_cfsr', 'hycom_ww3', 'hycom_ww3']
    basetitle = ['Ocean currents', 'Ocean currents & Stokes drift', 'Ocean currents & 3% wind',
                 None, 'Ocean currents & Stokes drift']
    
    if plot_f1 == True:
        figure1_overview(rivers=['solo', 'brantas', 'tanduy', 'citarum'],
                        cities=['Jakarta', 'Bandung', 'Surabaya', 'Surakarta'], output_path='plots/fig1.jpg', show=False)
    
    if plot_f2 == True:
        figure2_samples(output_path='plots/fig2.jpg', show=False)
    
    if plot_f3 == True or plot_f4 == True:
        for i in range(len(forcing)):
            if b[i] == None:
                description = forcing[i]
            else:
                description = f'{forcing[i]}_b{b[i]}_r{r[i]}'
                
            if basetitle[i] == None:
                title = ''
            else:
                if b[i] == None:
                    title = f'{basetitle[i]}, no beaching'
                else:
                    title = f'{basetitle[i]}\n Beaching $\lambda_b$ = {b[i]}, $\lambda_r$ = {r[i]}'
        
            input_path = f'{get_dir_from_json("pts_processed")}iot_particles_{description}.nc'
        
            ds_particles = xr.load_dataset(input_path)
        
            if plot_f3 == True:
                figure3_sources(ds_particles, title=title, output_path=f'plots/fig3_{description}.jpg', show=False)
            
            if plot_f4 == True:
                figure4_seasonality(ds_particles, rivers=['Solo', 'Brantas'],
                                    output_path=f'plots/fig4_{description}.jpg', show=False)
