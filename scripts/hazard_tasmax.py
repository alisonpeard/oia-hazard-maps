"""
Make hd35 return period maps for each model and scenario.
"""
# %%
import os
import shutil
import logging
from tqdm import tqdm
from glob import glob

import numpy as np
import xarray as xr
from scipy.stats import linregress

from stats import genextreme

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


input_dir = f"../data/yearly/tasmax"
scenarios = ["rcp85", "rcp45", "rcp26"]
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
epochs = [2080, 2050, 2030]
ignore = ["DS_Store"]


def plot_gev_fit(x, shape, loc, scale):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.hist(x, bins=20, density=True, alpha=0.6, color='g')
    x_grid = np.linspace(min(x), max(x), 100)
    pdf_fitted = genextreme.pdf(x_grid, shape, loc=loc, scale=scale)
    ax.plot(x_grid, pdf_fitted, 'r-', lw=2)
    ax.set_title(f"Fitted GEV at lat {lat}, lon {lon}\nn={len(x)}")
    return fig


if __name__ == "__main__":
    logging.basicConfig(filename='../logs/tasmax_ensemble.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    output_dir = "../data/hazard/tasmax/ensemble"
    os.makedirs(output_dir, exist_ok=True)
    
    figdir = os.path.join("../logs/gev_fits/")
    if os.path.exists(figdir):
        shutil.rmtree(figdir)
    os.makedirs(figdir, exist_ok=True)
    
    for scenario in scenarios:
        scenario_dir = os.path.join(input_dir, scenario)
        models = [d for d in os.listdir(scenario_dir) if d not in ignore]

        if len(models) == 0:
            logging.warning(f"No models found for scenario {scenario}")
            continue

        for model in models:
            model_dir = os.path.join(scenario_dir, model)
            files = glob(f"{model_dir}/*.nc")
            if len(files) == 0:
                logging.warning(f"No files found for model {model}, scenario {scenario}")
                continue

            ds = xr.open_mfdataset(files, engine="netcdf4")
            ds = ds.load()
            logging.info(f"Processing model: {model}, scenario: {scenario}")

            max_temp = ds["tasmax"].max().item()
            if max_temp > 60:
                logging.warning(f"Skipping model {model}, scenario {scenario}: max temp {max_temp}°C exceeds physical limits")
                continue

            ds_epoch = ds.copy()
            x = ds_epoch["tasmax"].values
            n, h, w = x.shape

            years = ds_epoch["year"].values
            years_centered = years - years.mean()

            # initialize arrays with NaNs
            tail_indices = np.full((h, w), np.nan)
            locs = np.full((h, w), np.nan)
            scales = np.full((h, w), np.nan)
            beta0_trends = np.full((h, w), np.nan)
            beta1_trends = np.full((h, w), np.nan)
            return_levels = np.full((len(rps), len(epochs), h, w), np.nan)

            # Define outfile HERE, outside the loops
            outfile = os.path.join(output_dir, scenario, model, "return_levels.nc")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            for lat in tqdm(range(h)):
                for lon in range(w):

                    data = x[:, lat, lon]
                    if (data == -9999).all() or np.isnan(data).all():
                        continue

                    # Create mask before filtering data
                    valid_mask = (data != -9999) & ~np.isnan(data)
                    years_valid = years_centered[valid_mask]
                    data_valid = data[valid_mask]

                    # fit a linear regression trend model using scipy
                    slope, intercept, r, p, se = linregress(years_valid, data_valid)

                    # check p-value
                    if p < 0.05:
                        logging.info(f"Significant trend at lat {lat}, lon {lon}: slope={slope}, p={p}")
                        beta0_trend = intercept
                        beta1_trend = slope
                    else:
                        logging.info(f"No significant trend at lat {lat}, lon {lon}: slope={slope}, p={p}")
                        beta0_trend = data_valid.mean()
                        beta1_trend = 0.0

                    beta0_trends[lat, lon] = beta0_trend
                    beta1_trends[lat, lon] = beta1_trend

                    expected_value = beta0_trend + beta1_trend * years_valid
                    residuals = data_valid - expected_value

                    shape, loc, scale = genextreme.fit(residuals)
                                       
                    tail_indices[lat, lon] = shape
                    scales[lat, lon] = scale
                    locs[lat, lon] = loc

                    random_number = np.random.rand()
                    if random_number < 0.001:
                        fig = plot_gev_fit(residuals, shape, loc, scale)
                        fig.savefig(os.path.join(figdir, f"{scenario}_{model}_lat{lat}_lon{lon}.png"))

                
                    # get return levels 
                    for i_epoch, epoch in enumerate(epochs):
                        epoch_centered = epoch - years.mean()
                        for rp_idx, T in enumerate(rps):
                            expected_value = beta0_trend + beta1_trend * epoch_centered
                            return_level = expected_value + genextreme.ppf(1 - 1 / T, shape, loc=loc, scale=scale)
                            if return_level > 60:
                                logging.warning(f"Unrealistic return level at lat {lat}, lon {lon}, epoch {epoch}, rp {T}: {return_level}°C")
                                return_level = np.nan
                            return_levels[rp_idx, i_epoch, lat, lon] = return_level
                    
        
            return_levels_da = xr.DataArray(
                return_levels,
                dims=["return_period", "epoch", "rlat", "rlon"],
                coords={
                    "return_period": rps,
                    "epoch": epochs,
                    "rlat": ds_epoch["rlat"],
                    "rlon": ds_epoch["rlon"],
                },
                name="tasmax",
                attrs={
                    "units": "°C",
                    "long_name": f"Annual maximum temperature°C",
                    "distribution": "Generalised Extreme Value Distribution (GEV)",
                },
            )

            # save results
            return_levels_ds = return_levels_da.to_dataset(name="tasmax")
            return_levels_ds["tail_index"] = (("rlat", "rlon"), tail_indices)
            return_levels_ds["scale"] = (("rlat", "rlon"), scales)
            return_levels_ds["threshold"] = (("rlat", "rlon"), locs)
            return_levels_ds.to_netcdf(outfile)
            logging.info(f"Saved GEV return levels to {outfile}")

            # view the latest one
            fig, axs = plt.subplots(
                1, 3, figsize=(25, 5),
                subplot_kw={"projection": ccrs.PlateCarree()}
            )

            vmin = return_levels_da.min().item()
            vmax = return_levels_da.max().item()

            return_levels_da.sel(epoch=2030, return_period=10).plot(cmap="YlOrRd", ax=axs[0], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2030, return_period=100).plot(cmap="YlOrRd", ax=axs[1], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2030, return_period=1000).plot(cmap="YlOrRd", ax=axs[2], 
                                                        vmin=vmin, vmax=vmax)

            for ax in axs:
                ax.add_feature(cfeature.COASTLINE)
                ax.set_title(ax.get_title(), fontsize=12)
            plt.show()
                
            logging.info(f"Finished processing GEV hazard maps for tasmax.")

    print(f"Finished processing all GEV hazard maps for tasmax.")
# %%
