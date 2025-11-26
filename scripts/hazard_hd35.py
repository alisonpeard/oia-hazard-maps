"""
Make hd35 return period maps for each model and scenario.
"""
# %%
import os
import logging
from glob import glob
from tqdm import tqdm

import numpy as np
import xarray as xr
from scipy.optimize import minimize

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from stats import binom
from stats import dgpd


input_dir = f"../data/yearly/hd35"
epochs = [2080, 2050, 2030]
scenarios = ["rcp85", "rcp45", "rcp26"]
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
ignore = ["DS_Store"]
q_threshold = 90
dry_run = False


if __name__ == "__main__":
    logging.basicConfig(filename='../logs/detrended.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    output_dir = f"..data/ensemble/hd35/q{q_threshold}"
    os.makedirs(output_dir, exist_ok=True)
    
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

            ds_epoch = ds.copy()
            x = ds_epoch[f"hd35"].values
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

            pixel_count = 0  # Renamed from 'i' to avoid confusion


            for lat in tqdm(range(h)):
                for lon in range(w):

                    data = x[:, lat, lon]
                    if (data == -9999).all() or np.isnan(data).all():
                        continue
                    pixel_count += 1

                    # Create mask before filtering data
                    valid_mask = (data != -9999) & ~np.isnan(data)
                    years_valid = years_centered[valid_mask]
                    data_valid = data[valid_mask]

                    # estimate initial parameters for trend model
                    p_init = np.clip(data_valid.mean() / 365, 0.01, 0.99)  # avoid extremes
                    beta0_init = np.log(p_init / (1 - p_init))  # logit of mean probability

                    # fit the trend model
                    fit_trend = minimize(
                        binom.nll,
                        [beta0_init, 0],
                        args=(data_valid, years_valid),
                        method='L-BFGS-B',
                        bounds=[(-10, 10), (-0.1, 0.1)]
                    )

                    if not fit_trend.success:
                        logging.warning(f"Trend fitting failed at lat {lat}, lon {lon}: {fit_trend.message}")
                        continue

                    beta0_trend, beta1_trend = fit_trend.x
                    logging.info(f"Intercept: {beta0_trend:.4f}, Slope: {beta1_trend:.4f}")
                    
                    mean_rates = binom.expected_value(years_valid, (beta0_trend, beta1_trend))
                    residuals = data_valid - mean_rates
                    
                    threshold = np.percentile(residuals, q_threshold)
                    logging.info(f"\nChosen threshold at lat {lat}, lon {lon}: {threshold}")

                    exceedances = residuals[residuals > threshold] - threshold
                    exceed_years = years_valid[residuals > threshold]

                    if len(exceedances) < 10:
                        logging.warning(f"Not enough exceedances at lat {lat}, lon {lon}")
                        continue
                    logging.info(f"Number of exceedances at lat {lat}, lon {lon}: {len(exceedances)}\n")
                    
     
                    scale_init, shape_init = dgpd.guess_params(exceedances)
                    init_params = [scale_init, shape_init]
                    
                    if scale_init <= 0:
                        logging.warning(f"Initial guess for scale is non-positive at lat {lat}, lon {lon}")
                        continue

                    result = minimize(
                        dgpd.nll,
                        init_params,
                        args=(exceedances,),
                        method='L-BFGS-B',
                        bounds=[(2e-5, None), (-0.5, 0.5)]
                    )
                    
                    if not result.success:
                        logging.warning(f"Optimization failed at lat {lat}, lon {lon}: {result.message}")
                        continue
                    
                    scale, shape = result.x                        
                    tail_indices[lat, lon] = shape
                    scales[lat, lon] = scale
                    locs[lat, lon] = threshold
                
                    # get return levels 
                    for i_epoch, epoch in enumerate(epochs):
                        epoch_centered = epoch - years.mean()
                        for rp_idx, T in enumerate(rps):
                            base_rate = binom.expected_value(epoch_centered, (beta0_trend, beta1_trend))
                            return_level = base_rate + threshold + dgpd.quantile(1/T, scale, shape)
                            return_levels[rp_idx, i_epoch, lat, lon] = min(return_level, 365)
                    
        
            return_levels_da = xr.DataArray(
                return_levels,
                dims=["return_period", "epoch", "rlat", "rlon"],
                coords={
                    "return_period": rps,
                    "epoch": epochs,
                    "rlat": ds_epoch["rlat"],
                    "rlon": ds_epoch["rlon"],
                },
                name=f"hd35",
                attrs={
                    "units": "days",
                    "long_name": f"Return levels for annual number of days exceeding 35°C",
                    "distribution": "Discrete Generalized Pareto Distribution (D-GPD)",
                },
            )

            # save results
            return_levels_ds = return_levels_da.to_dataset(name=f"hd35")
            return_levels_ds["tail_index"] = (("rlat", "rlon"), tail_indices)
            return_levels_ds["scale"] = (("rlat", "rlon"), scales)
            return_levels_ds["threshold"] = (("rlat", "rlon"), locs)


            return_levels_ds.to_netcdf(outfile)
            logging.info(f"Saved D-GPD return levels to {outfile}")

            # view the latest one
            fig, axs = plt.subplots(1, 3, figsize=(25, 5), 
                                    subplot_kw={"projection": ccrs.PlateCarree()})

            vmin = return_levels_da.min().item()
            vmax = return_levels_da.max().item()

            return_levels_da.sel(epoch=2080, return_period=10).plot(cmap="YlOrRd", ax=axs[0], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2080, return_period=100).plot(cmap="YlOrRd", ax=axs[1], 
                                                        vmin=vmin, vmax=vmax)
            return_levels_da.sel(epoch=2080, return_period=1000).plot(cmap="YlOrRd", ax=axs[2], 
                                                        vmin=vmin, vmax=vmax)

            for ax in axs:
                ax.add_feature(cfeature.COASTLINE)
                ax.set_title(ax.get_title(), fontsize=12)
            plt.show()
                
            logging.info(f"Finished processing d-GPD hazard maps for hd35.")

    print(f"Finished processing d-GPD hazard maps for hd35 and threshold q{q_threshold}.")
# %%
