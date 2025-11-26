# %%
import os
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt


temperature = 35  # temperature threshold for hd35
scenarios = ["rcp26", "rcp45", "rcp85"]
thresholds = ["q90"] #["q70", "q80", "q90"]
epochs = [2030, 2050, 2080]
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
wd = f"../data/hazard/hd{temperature}/ensemble"
outdir = f"../data/hazard/hd{temperature}/tifs"

# delete and recreate output dir
if os.path.exists(outdir):
    import shutil
    shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)
dry_run = False

for scenario in scenarios:
    hazard_maps = []
    template_ds = None
    for threshold in thresholds:
        indir = os.path.join(wd, threshold, scenario)
        models = os.listdir(indir)
        print(f"{models=}")
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
            if dry_run:
                ds.isel(return_period=0).hd35.plot(cmap="YlOrRd")
                plt.show()
            hazard_maps.append(ds[f"hd{temperature}"])

    if hazard_maps:
        print(f"Combining {len(hazard_maps)} hazard maps for scenario {scenario}")
        combined = xr.concat(hazard_maps, dim="model")
        haz_mean = combined.mean(dim="model", skipna=True)
        haz_min = combined.min(dim="model", skipna=True)
        haz_max = combined.max(dim="model", skipna=True)

    for epoch in epochs:
        for rp in rps:
            haz_mean = combined.sel(epoch=epoch, return_period=rp).mean(dim="model", skipna=True)
            haz_min = combined.sel(epoch=epoch, return_period=rp).min(dim="model", skipna=True)
            haz_max = combined.sel(epoch=epoch, return_period=rp).max(dim="model", skipna=True)

            # clip to (0, 365)
            haz_mean = haz_mean.clip(0, 365)
            haz_min = haz_min.clip(0, 365)
            haz_max = haz_max.clip(0, 365)

            haz_mean.rio.write_crs("EPSG:4326", inplace=True)
            haz_min.rio.write_crs("EPSG:4326", inplace=True)
            haz_max.rio.write_crs("EPSG:4326", inplace=True)

            outmean = f"hd{temperature}_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"
            outmin = f"hd{temperature}min_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"
            outmax = f"hd{temperature}max_{epoch}_{scenario}_rp{str(rp).zfill(5)}.tif"

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

sample_file = os.path.join(outdir, f"hd{temperature}_2030_rcp26_rp00005.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}min_")
    max_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}max_")
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
    fig.suptitle(f"HD35 Hazard Maps for 2030 RCP26 RP5 ({threshold})")

sample_file = os.path.join(outdir, f"hd{temperature}_2080_rcp85_rp01000.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}min_")
    max_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}max_")
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
    fig.suptitle(f"HD35 Hazard Maps for 2080 RCP85 RP1000 ({threshold})")

sample_file = os.path.join(outdir, f"hd{temperature}_2050_rcp45_rp00100.tif")
if os.path.exists(sample_file):
    min_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}min_")
    max_file = sample_file.replace(f"hd{temperature}_", f"hd{temperature}max_")
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
    fig.suptitle(f"HD35 Hazard Maps for 2050 RCP45 RP100 ({threshold})")
# %%
