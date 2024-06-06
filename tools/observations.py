from tools.files import get_dir_from_json
from tools.land import get_island_boxes_from_toml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

cki, ci = get_island_boxes_from_toml()

def read_iot_cleanup_events(input_path=get_dir_from_json('iot_measurements')):
    df = pd.read_excel(input_path, sheet_name='Event details', skiprows=5)
    time = np.array([d.to_pydatetime() for d in df['Date']])
    lon = np.array(df['Longitude'])
    lat = np.array(df['latitude'])
    n_plastic = np.array(df['Total'])
    kg_plastic = np.array(df['Weight Kg'])
    return time, lon, lat, n_plastic, kg_plastic

def _read_cleanup_events_island(island:dict, input_path=get_dir_from_json('iot_measurements')):
    time, lon, lat, n_plastic, kg_plastic = read_iot_cleanup_events(input_path=input_path)
    l_lon = np.logical_and(island['lon_range'][0] <= lon, lon <= island['lon_range'][1])
    l_lat = np.logical_and(island['lat_range'][0] <= lat, lat <= island['lat_range'][1])
    l_island = np.logical_and(l_lon, l_lat)
    return time[l_island], lon[l_island], lat[l_island], n_plastic[l_island], kg_plastic[l_island]

def read_ci_cleanup_events(input_path=get_dir_from_json('iot_measurements')):
    time, lon, lat, n_plastic, kg_plastic = _read_cleanup_events_island(ci, input_path)
    return time, lon, lat, n_plastic, kg_plastic

def read_cki_cleanup_events(input_path=get_dir_from_json('iot_measurements')):
    time, lon, lat, n_plastic, kg_plastic = _read_cleanup_events_island(cki, input_path)
    return time, lon, lat, n_plastic, kg_plastic

def get_monthly_plastic_samples(plastic_type='count') -> tuple:
    if not np.logical_or(plastic_type=='count', plastic_type=='mass'):
        raise ValueError(f'Unknown plastic type requested: {plastic_type}. Valid values are: count and mass.')

    months = np.arange(1,13,1)
    # plastic measurements
    time_cki, _, _, n_plastic_cki, kg_plastic_cki = read_cki_cleanup_events()
    time_ci, _, _, n_plastic_ci, kg_plastic_ci = read_ci_cleanup_events()
    n_plastic_month_cki = np.zeros(12)
    n_months_cki = np.zeros(12)
    n_plastic_month_ci = np.zeros(12)
    n_months_ci = np.zeros(12)
    # cki
    for i, t in enumerate(time_cki):
        n_months_cki[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], n_plastic_cki[i]])
        elif plastic_type == 'mass':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], kg_plastic_cki[i]])
    # ci
    for i, t in enumerate(time_ci):
        n_months_ci[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], n_plastic_ci[i]])
        elif plastic_type == 'mass':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], kg_plastic_ci[i]])
    
    return (months, n_plastic_month_cki, n_plastic_month_ci, n_months_cki, n_months_ci)
    
def read_iot_plastic_type_counts(input_path=get_dir_from_json('iot_measurements')) -> tuple:
    df = pd.read_excel(input_path, sheet_name='Item lists', skiprows=4)
    material_counts = pd.value_counts(df['Material'])
    df_plastic = df[df['Material']=='Plastic']
    category_counts = pd.value_counts(df_plastic['Datasheet category'])
    df_plastic_land = df_plastic[df_plastic['Datasheet category'] != 'Plastic Fishing Items']
    item_counts = pd.value_counts(df_plastic_land['Item'])
    
    return material_counts, category_counts, item_counts
