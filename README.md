# iot_plastics
Determining sources of plastic waste on the Indian Ocean Territories

## Set-up
Create a "input/dirs.json" file with the paths as specified in "input/example_dirs.json".

Create conda environment using the environment.yml file.

## Particle tracking simulations
Simulations are run in the `pts_parcels.py` script. Particles are released from river plastic sources as determined by Lebreton et al. (2017) (original file in input/PlasticRiverInputs_Lebreton2017.shp, file with sources moved to HYCOM ocean grid cells in input/PlasticRiverSources_Lebreton2017_Hycom.nc), with 1 particle representing 1 tonne of plastic debris. Particles are released at the start of every month for 2008 and simulations are run until the end of 2009. Particles are transported by HYCOM ocean surface currents and WaveWatch III surface Stokes drift.

## Determining sources
Sources of particles reaching the islands (defined in "input/islands.toml") are determined in the `processing.py` script.
