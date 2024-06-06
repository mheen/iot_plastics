from plastic_sources import get_iot_sources
from parcels_kernels import AdvectionRK4_Beaching, DiffusionUniformKh_Beaching, OninkBeachingKernel, OninkResusKernel, BorderKernel, delete_particle
from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools.land import LandMask
from parcels import Field, FieldSet
from parcels import JITParticle, Variable, ErrorCode, FieldSet, ParticleSet
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

def _get_indices_from_landmask(lon_range, lat_range, input_path='input/hycom_landmask.nc'):
    lm = xr.load_dataset(input_path)
    i_lon_start = list(lm.lon.values).index(lm.sel(lon=lon_range[0], method='nearest').lon)
    i_lon_end = list(lm.lon.values).index(lm.sel(lon=lon_range[1], method='nearest').lon)
    i_lat_start = list(lm.lat.values).index(lm.sel(lat=lat_range[0], method='nearest').lat)
    i_lat_end = list(lm.lat.values).index(lm.sel(lat=lat_range[1], method='nearest').lat)
    indices = {'lon' : range(i_lon_start,i_lon_end), 'lat': range(i_lat_start,i_lat_end)}
    return indices

class IOTRiverParticles:
    def __init__(self,lon0,lat0,time0):
        self.lon0 = lon0
        self.lat0 = lat0
        self.time0 = time0

    @staticmethod
    def get_from_netcdf(start_date):
        iot_sources = get_iot_sources()
        year = start_date.year
        day = start_date.day
        hour = start_date.hour
        minute = start_date.minute
        lon0 = []
        lat0 = []
        time0 = []
        for t in iot_sources.time.astype('int'):
            time = datetime(year,t,day,hour,minute)
            lon0_temp, lat0_temp, time0_temp = iot_sources.convert_for_parcels_per_time(t-1,time)
            lon0 = np.append(lon0, lon0_temp)
            lat0 = np.append(lat0, lat0_temp)
            time0 = np.append(time0, time0_temp)
        return IOTRiverParticles(lon0, lat0, time0)

class BeachingParticle(JITParticle):
    #Now the beaching variables
    #0=open ocean, 1=beached
    beached = Variable(
        'beached', dtype=np.int32,
        initial=0,
    )
    # Land value at the particle's location
    # 0 -> ocean
    # 1 -> land
    # 0 < land_value < 1 -> beaching_region
    land_value = Variable(
        'land_value', dtype=np.float32,
        initial=0.0,
    )

    previous_lon = Variable(
        'previous_lon', dtype=np.float32,
        initial=0.0,
        to_write=False,
    )
    previous_lat = Variable(
        'previous_lat', dtype=np.float32,
        initial=0.0,
        to_write=False,
    )

def create_fset(input_dir:str, start_date:datetime, end_date:datetime,
                lon_range:list[float], lat_range:list[float],
                input_path_lm:str) -> FieldSet:
    variables = {'U':'u','V':'v'}
    dimensions = {'lat':'lat','lon':'lon','time':'time'}
    ncfiles = get_daily_files_in_time_range(input_dir, start_date, end_date, 'nc')
    indices = _get_indices_from_landmask(lon_range, lat_range, input_path=input_path_lm)
    fset = FieldSet.from_netcdf(ncfiles, variables, dimensions, indices=indices)
    return fset

def create_fset_currents_stokes(input_dir_c:str, input_dir_s:str, start_date:datetime, end_date:datetime,
                                lon_range:list[float], lat_range:list[float]) -> FieldSet:
    fset_c = create_fset(input_dir_c, start_date, end_date, lon_range, lat_range, 'input/hycom_landmask.nc')
    fset_s = create_fset(input_dir_s, start_date, end_date, lon_range, lat_range, f'{input_dir_s}{start_date.strftime("%Y%m%d")}.nc')

    fset = FieldSet(U=fset_c.U + fset_s.U, V=fset_c.V + fset_s.V)
    return fset

def create_fset_currents_wind(input_dir_c:str, input_dir_w:str, windage:float,
                              start_date:datetime, end_date:datetime,
                              lon_range:list[float], lat_range:list[float]) -> FieldSet:    
    fset_c = create_fset(input_dir_c, start_date, end_date, lon_range, lat_range, 'input/hycom_landmask.nc')
    
    fset_w = create_fset(input_dir_w, start_date, end_date, lon_range, lat_range, f'{input_dir_w}{start_date.strftime("%Y%m%d")}.nc')
    fset_w.U.set_scaling_factor(windage)
    fset_w.V.set_scaling_factor(windage)
    
    fset = FieldSet(U=fset_c.U + fset_w.U, V=fset_c.V + fset_w.V)
    return fset

def run_parcels(fset:FieldSet, start_date:datetime, end_date:datetime,
                beaching_timescale_days:float,
                resuspension_timescale_days:float,
                output_path:str,
                kh = 10.,
                dt = timedelta(hours=1),
                output_interval = 24):
    
    # add constant horizontal diffusivity (zero on land)
    lm = LandMask.read_from_netcdf()
    lm.extend_iot_land()
    kh2D = kh*np.ones(lm.mask.shape)
    kh2D[lm.mask.astype('bool')] = 0.0 # diffusion zero on land  
    fset.add_field(Field('Kh_zonal', data=kh2D, lon=lm.lon, lat=lm.lat, mesh='spherical', interp_method='linear'))
    fset.add_field(Field('Kh_meridional', data=kh2D, lon=lm.lon, lat=lm.lat, mesh='spherical', interp_method='linear'))
    
    # add land field
    fset.add_field(Field('land', data=lm.mask, lon=lm.lon, lat=lm.lat, mesh='spherical', interp_method='linear'))
    
    # add beaching and resuspension timescales
    if beaching_timescale_days != None:
        seconds_b = beaching_timescale_days * (24*60*60)
        fset.add_constant('beaching_constant', seconds_b)
    if resuspension_timescale_days != None:
        seconds_r = resuspension_timescale_days * (24*60*60)
        fset.add_constant('resuspension_constant', seconds_r)
    
    # get releases
    iot_sources = IOTRiverParticles.get_from_netcdf(start_date)
    lon0 = iot_sources.lon0
    lat0 = iot_sources.lat0
    time0 = iot_sources.time0
    
    # create particleset
    pset = ParticleSet(fieldset=fset,
                       pclass=BeachingParticle,
                       lon=lon0,
                       lat=lat0,
                       time=time0)
    
    # set kernels
    kernels = pset.Kernel(AdvectionRK4_Beaching) + pset.Kernel(DiffusionUniformKh_Beaching) + pset.Kernel(BorderKernel)
    if beaching_timescale_days != None:
        kernels = kernels + pset.Kernel(OninkBeachingKernel)
    if resuspension_timescale_days != None:
        kernels = kernels + pset.Kernel(OninkResusKernel)
    
    # run simulation
    output_file = pset.ParticleFile(name=output_path, outputdt=dt*output_interval)
    pset.execute(kernels,
                 runtime=timedelta(days=(end_date-start_date).days),
                 dt=dt,
                 output_file=output_file,
                 verbose_progress=True,
                 recovery={ErrorCode.ErrorOutOfBounds: delete_particle})
    output_file.close()

if __name__ == '__main__':
    input_dir = get_dir_from_json('hycom_input')
    input_dir_s = get_dir_from_json('ww3_input')
    
    beaching_timescale_days = 10 # Onink et al: 100 days as max, here using [10, 100]
    resuspension_timescale_days = 70 # Hinanta et al: between 69-273 days (also used by Onink et al), here using [70, 270]
    
    output_path = f'{get_dir_from_json("pts_output")}iot_hycom_ww3_b10_r70.nc'
    
    start_date = datetime(2008, 1, 1)
    end_date = datetime(2009, 12, 31)   
    lon_range = [90., 140.]
    lat_range = [-20., 20.]
    
    fset = create_fset_currents_stokes(input_dir, input_dir_s, start_date, end_date, lon_range, lat_range)
    
    run_parcels(fset, start_date, end_date, beaching_timescale_days, resuspension_timescale_days, output_path)  
