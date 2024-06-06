import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def add_subtitle(ax:plt.axes, subtitle:str, location='upper left') -> plt.axes:
    anchored_text = AnchoredText(subtitle, loc=location, borderpad=0.0)
    anchored_text.zorder = 15
    ax.add_artist(anchored_text)
    return ax

def color_y_axis(ax:plt.axes, color:str, spine_location:str):
    ax.spines[spine_location].set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    return ax

def plot_box(ax:plt.axes, lon_range:list, lat_range:list,
             style='-', width=0.7, color='k'):
    x = [lon_range[0], lon_range[1], lon_range[1], lon_range[0], lon_range[0]]
    y = [lat_range[0], lat_range[0], lat_range[1], lat_range[1], lat_range[0]]
    
    ax.plot(x, y, linewidth=width, linestyle=style, color=color)