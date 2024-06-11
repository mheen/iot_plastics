import parcels.rng as ParcelsRandom
import math

def delete_particle(particle, fieldset, time):
    particle.delete()

def AdvectionRK4_Beaching(particle, fieldset, time):
    # Code from https://github.com/VeckoTheGecko/ocean-plastic-honours
    if particle.beached == 0.0:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

def DiffusionUniformKh_Beaching(particle, fieldset, time):
    # Code from https://github.com/VeckoTheGecko/ocean-plastic-honours
    if particle.beached == 0.0:
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy

# Beaching kernels
def OninkBeachingKernel(particle, fieldset, time):
    """
    Beaches the particle if it is within one gridcell of the land.
    """
    # Code from https://github.com/VeckoTheGecko/ocean-plastic-honours
    particle.land_value = fieldset.land[particle] # ie. if land = 0, in ocean. If 0 < land < 1, in beaching region. If land = 1, on land (collided with coast).
    proba_beach = fieldset.beaching_constant
    proba_dt = 1.0 - math.exp(- particle.dt / proba_beach) # Converting to probability to apply for each timestep
    if particle.beached == 0.0 and fieldset.land[particle] > 0.0 and ParcelsRandom.random() < proba_dt: # ie. particle is floating and is in beaching region
        particle.beached = 1.0

def OninkResusKernel(particle, fieldset, time):
    """
    Beaches the particle if it is within one gridcell of the land.
    """
    # Code from https://github.com/VeckoTheGecko/ocean-plastic-honours
    proba_resus = fieldset.resuspension_constant
    proba_dt = 1.0 - math.exp(- particle.dt / proba_resus) # Converting to probability to apply for each timestep
    if particle.beached == 1.0 and ParcelsRandom.random() < proba_dt: # ie. particle is beached
        particle.beached = 0.0

def BorderKernel(particle, fieldset, time):
    """
    If a particle directly is close to land, it gets a nudge out to the ocean
    """
    # Code from https://github.com/VeckoTheGecko/ocean-plastic-honours
    if fieldset.land[particle] > 0.9:
        right = math.floor(fieldset.land[time, particle.depth, particle.lat, particle.lon + 1_000])
        left = math.floor(fieldset.land[time, particle.depth, particle.lat, particle.lon - 1_000])
        up = math.floor(fieldset.land[time, particle.depth, particle.lat + 1_000, particle.lon])
        down = math.floor(fieldset.land[time, particle.depth, particle.lat - 1_000, particle.lon])
        x = -(right - left) # Move opposite to direction of land
        y = -(up - down)
        x = x / math.sqrt(x**2 + y**2)
        y = y / math.sqrt(x**2 + y**2)
        particle.lon = particle.lon + x * particle.dt # 1m/s nudge for dt out to sea
        particle.lat = particle.lat + y * particle.dt
