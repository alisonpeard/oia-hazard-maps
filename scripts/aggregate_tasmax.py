# %%
import os
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt


scenarios = ["historical", "rcp26", "rcp45", "rcp85"]
epochs = {
    "historical": [2010],
    "rcp26": [2030, 2050, 2080],
    "rcp45": [2030, 2050, 2080],
    "rcp85": [2030, 2050, 2080]
}
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
wd = f"../data/hazard/tasmax/ensemble"
outdir = f"../data/hazard/tasmax/tifs"
shared_models = ['MOHC-HadGEM2-ES_KNMI-RACMO22T', 'MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009']

# %% delete and recreate output dir
if os.path.exists(outdir):
    import shutil
    shutil.rmtree(outdir,  ignore_errors=True)
os.makedirs(outdir, exist_ok=True)
plot = True

for scenario in scenarios:
    template_ds = None

    indir = os.path.join(wd, scenario)
    models = shared_models # os.listdir(indir)
    print(f"{models=}")

    hazard_maps = []
    for model in models:
        filepath = os.path.join(indir, model, "return_levels.nc")
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            continue
        ds = xr.open_dataset(filepath, engine="netcdf4", decode_times=False)

        if template_ds is None:
            template_ds = ds.copy()
        else:
            ds = ds.interp(rlat=template_ds.rlat, rlon=template_ds.rlon, method='nearest')
        if plot:
            ds.isel(epoch=0, return_period=0).tasmax.plot(cmap="YlOrRd")
            plt.show()
        hazard_maps.append(ds["tasmax"])

    if hazard_maps:
        print(f"Combining {len(hazard_maps)} hazard maps for scenario {scenario}")
        combined = xr.concat(hazard_maps, dim="model")
        haz_mean = combined.mean(dim="model", skipna=True)
        haz_min = combined.min(dim="model", skipna=True)
        haz_max = combined.max(dim="model", skipna=True)

    for epoch in epochs[scenario]:
        for rp in rps:
            haz_mean = combined.sel(epoch=epoch, return_period=rp).mean(dim="model", skipna=True)
            haz_min = combined.sel(epoch=epoch, return_period=rp).min(dim="model", skipna=True)
            haz_max = combined.sel(epoch=epoch, return_period=rp).max(dim="model", skipna=True)

            haz_mean.rio.write_crs("EPSG:4326", inplace=True)
            haz_min.rio.write_crs("EPSG:4326", inplace=True)
            haz_max.rio.write_crs("EPSG:4326", inplace=True)

            outmean = f"tasmax_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"
            outmin = f"tasmaxmin_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"
            outmax = f"tasmaxmax_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"

            outmeanpath = os.path.join(outdir, outmean)
            outminpath = os.path.join(outdir, outmin)
            outmaxpath = os.path.join(outdir, outmax)

            haz_mean.rio.to_raster(outmeanpath)
            haz_min.rio.to_raster(outminpath)
            haz_max.rio.to_raster(outmaxpath)

            print(f"Saved ensemble mean hazard map to {outmeanpath}")
            print(f"Saved ensemble min hazard map to {outminpath}")
            print(f"Saved ensemble max hazard map to {outmaxpath}")

print("Finished exporting hazard maps to GeoTIFFs.")
# %%
            
# load and plot one of the generated files
import rioxarray

sample_file = os.path.join(outdir, "tasmax_2010_historical_rp00005.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace("tasmax_", "tasmaxmin_")
    max_file = sample_file.replace("tasmax_", "tasmaxmax_")
    ds_min = rioxarray.open_rasterio(min_file)
    ds_max = rioxarray.open_rasterio(max_file)

    da_range = ds_max - ds_min

    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5))
    ds_min.plot(ax=axs[0], cmap="YlOrRd")
    ds_max.plot(ax=axs[1], cmap="YlOrRd")
    da_range.plot(cmap="YlOrRd", ax=axs[2])
    axs[0].set_title("Minimum HD35")
    axs[1].set_title("Maximum HD35")
    axs[2].set_title("Range HD35")
    fig.suptitle(f"HD35 Hazard Maps for 2030 RCP26 RP5")

sample_file = os.path.join(outdir, "tasmax_2080_rcp85_rp01000.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace("tasmax_", "tasmaxmin_")
    max_file = sample_file.replace("tasmax_", "tasmaxmax_")
    ds_min = rioxarray.open_rasterio(min_file)
    ds_max = rioxarray.open_rasterio(max_file)

    da_range = ds_max - ds_min

    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5))
    ds_min.plot(ax=axs[0], cmap="YlOrRd")
    ds_max.plot(ax=axs[1], cmap="YlOrRd")
    da_range.plot(cmap="YlOrRd", ax=axs[2])
    axs[0].set_title("Minimum HD35")
    axs[1].set_title("Maximum HD35")
    axs[2].set_title("Range HD35")
    fig.suptitle(f"HD35 Hazard Maps for 2080 RCP85 RP1000")

sample_file = os.path.join(outdir, "tasmax_2050_rcp45_rp00100.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace("tasmax_", "tasmaxmin_")
    max_file = sample_file.replace("tasmax_", "tasmaxmax_")
    ds_min = rioxarray.open_rasterio(min_file)
    ds_max = rioxarray.open_rasterio(max_file)

    da_range = ds_max - ds_min

    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5))
    ds_min.plot(ax=axs[0], cmap="YlOrRd")
    ds_max.plot(ax=axs[1], cmap="YlOrRd")
    da_range.plot(cmap="YlOrRd")

    axs[0].set_title("Minimum HD35")
    axs[1].set_title("Maximum HD35")
    axs[2].set_title("Range HD35")
    fig.suptitle(f"HD35 Hazard Maps for 2050 RCP45 RP100")
# %%
