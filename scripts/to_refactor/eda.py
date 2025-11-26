"""
Use this to examine rolling statistics over the time series
"""
# %%
import os
from glob import glob
import random
import matplotlib.pyplot as plt
import xarray as xr

datadir = "annual_stats"
scenario = "rcp45"
model = "MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17"
var = "hd39"
indir = os.path.join(datadir, scenario, model)
files = sorted(glob(os.path.join(indir, "*.nc")))

ds = xr.open_mfdataset(files,  engine="netcdf4", decode_times=False)
ds.load()
# %%
nlat = ds["rlat"].size
nlon = ds["rlon"].size


# set seed
random.seed(42)
fig, axs = plt.subplots(5, 4, sharex=True, figsize=(15,10))

for ax in axs.flat:
    allzero = True
    while allzero:
        i = random.randint(0, nlat - 1)
        j = random.randint(0, nlon - 1)

        ts = ds[var].isel(rlat=i, rlon=j)
        if ts.sum().item() > 0:
            allzero = False

    ts.plot(ax=ax)
    ax.set_title(f"lat: {ds['rlat'].isel(rlat=i).item():.2f}, lon: {ds['rlon'].isel(rlon=j).item():.2f}")

fig.suptitle(f"{var} time series for {model} under {scenario}")
# %%
