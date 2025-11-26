# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import genpareto
from scipy.optimize import minimize
import logging
from tqdm import tqdm


def dgpd_pmf(k, scale, shape):
    """
    Discrete Generalized Pareto Distribution PMF
    k: integer values (0, 1, 2, ...)
    scale: scale parameter (σ)
    shape: shape parameter (ξ)

    From https://arxiv.org/pdf/1707.05033
    """
    if shape == 0:
        # top of page 4
        p = 1 - np.exp(-1/scale)
        return p * np.exp(-k/scale)
    else:
        # Eq. (4)
        k_lower = 1 + shape * k / scale
        k1_lower = 1 + shape * (k + 1) / scale
        if k_lower <= 0 or k1_lower <= 0:
            return 0.0
        
        gpd_k = (1 + shape * k / scale)**(-1/shape)
        gpd_k1 = (1 + shape * (k + 1) / scale)**(-1/shape)
        return gpd_k - gpd_k1


def dgpd_survival(k, scale, shape):
    """Survival function for D-GPD. This is the same 
    as the GPD."""
    if shape == 0:
        return np.exp(-k/scale)
    else:
        return (1 + shape * k / scale)**(-1/shape)


def dgpd_quantile(p, scale, shape):
    """
    Quantile function for D-GPD (return level)
    p: exceedance probability (e.g., 1/return_period)
    """
    if shape == 0:
        x = -scale * np.log(p)
    else:
        x = (scale / shape) * (p**(-shape) - 1)
    return np.floor(x) # only floor if needd


def nll_dgpd(params, data):
    """Negative log-likelihood for D-GPD"""
    scale, shape = params
    
    if scale <= 0:
        return np.inf
    
    log_probs = []
    for k in data:
        pmf_val = dgpd_pmf(k, scale, shape)
        if pmf_val <= 0:
            return np.inf
        log_probs.append(np.log(pmf_val))
    
    return -np.sum(log_probs)


window = 30
temperatures = [35, 39]
input_dir = "annual_stats"
scenarios = ["rcp26", "rcp45", "rcp85"]
rps = [5, 10, 20, 50, 100, 200, 500, 1000]
epochs = [2030, 2050, 2080]
ignore = ["DS_Store"]
dry_run = True


if __name__ == "__main__":
    # configure logging to log file
    logging.basicConfig(filename='./d-gpd.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    for temperature in temperatures:
        output_dir = f"hazard_model/hd{temperature}"
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario in scenarios:
            scenario_dir = os.path.join(input_dir, scenario)
            models = [d for d in os.listdir(scenario_dir) if d not in ignore]

            if len(models) == 0:
                logging.warning(f"No models found for scenario {scenario}")
                continue

            for model in (pbar := tqdm(models)):
                pbar.set_description(f"Processing model {model}, scenario {scenario}, hd{temperature}°C")

                model_dir = os.path.join(scenario_dir, model)
                files = glob(f"{model_dir}/*.nc")
                if len(files) == 0:
                    logging.warning(f"No files found for model {model}, scenario {scenario}")
                    continue

                ds = xr.open_mfdataset(files, engine="netcdf4")
                ds = ds.load()

                logging.info(f"Processing model: {model}, scenario: {scenario}")

                for epoch in epochs:
                    outfile = os.path.join(output_dir, model, scenario, f"{epoch}.nc")
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                    ds_epoch = ds.sel(year=slice(epoch - window/2, epoch + window/2 - 1))
                    ds_epoch = ds_epoch.load()

                    x = ds_epoch[f"hd{temperature}"].values
                    n, h, w = x.shape

                    years = ds_epoch["year"].values

                    # initialize arrays with NaNs
                    tail_indices = np.full((h, w), np.nan)
                    locs = np.full((h, w), np.nan)
                    scales = np.full((h, w), np.nan)
                    return_levels = np.full((len(rps), h, w), np.nan)

                    i = 0

                    for lat in range(h):
                        for lon in range(w):

                            data = x[:, lat, lon]
                            if (data == -9999).all() or np.isnan(data).all():
                                continue
                            
                            valid_mask = (data != -9999) & (~np.isnan(data))
                            data = data[valid_mask]
                            years_valid = years[valid_mask]
                        
                            residuals = data
                            threshold = 0.0
                            
                            logging.info(f"Chosen threshold at lat {lat}, lon {lon}: {threshold}")
                            exceedances = residuals[residuals > threshold] - threshold
                            exceed_years = years_valid[residuals > threshold]

                            if len(exceedances) < 20:
                                logging.warning(f"Not enough exceedances at lat {lat}, lon {lon}")
                                continue
                            else:
                                logging.info(f"Number of exceedances at lat {lat}, lon {lon}: {len(exceedances)}")
                            
                            # initial estimates using normal GPD
                            init_params = genpareto.fit(exceedances, floc=0)
                            scale_init = init_params[2]
                            shape_init = init_params[0]
                            init_params = [scale_init, shape_init] # for stationary
                            
                            if scale_init <= 0:
                                logging.warning(f"Initial guess for scale is non-positive at lat {lat}, lon {lon}")
                                continue

                            result = minimize(
                                nll_dgpd,
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
                            for i, T in enumerate(rps):
                                base_rate = 0.0
                                return_level = base_rate + threshold + dgpd_quantile(1/T, scale, shape)
                                logging.info(f"Return level for T={T} at lat {lat}, lon {lon}, epoch {epoch}: {return_level}")
                                return_levels[i, lat, lon] = min(return_level, 365)

                            i += 1

                    # create xarray DataArray for all pixels
                    return_levels_da = xr.DataArray(
                        return_levels,
                        dims=["return_period", "rlat", "rlon"],
                        coords={
                            "return_period": rps,
                            "rlat": ds_epoch["rlat"],
                            "rlon": ds_epoch["rlon"],
                        },
                        name=f"hd{temperature}",
                        attrs={
                            "units": "days",
                            "long_name": f"Return levels for annual number of days exceeding {temperature}°C",
                            "distribution": "Discrete Generalized Pareto Distribution (D-GPD)",
                        },
                    )

                    # save results
                    return_levels_ds = return_levels_da.to_dataset(name=f"hd{temperature}")
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

                    return_levels_da.sel(return_period=10).plot(cmap="YlOrRd", ax=axs[0], 
                                                                vmin=vmin, vmax=vmax)
                    return_levels_da.sel(return_period=100).plot(cmap="YlOrRd", ax=axs[1], 
                                                                vmin=vmin, vmax=vmax)
                    return_levels_da.sel(return_period=1000).plot(cmap="YlOrRd", ax=axs[2], 
                                                                vmin=vmin, vmax=vmax)

                    for ax in axs:
                        ax.add_feature(cfeature.COASTLINE)
                        ax.set_title(ax.get_title(), fontsize=12)
                    plt.show()
                        
                    logging.info(f"Finished processing d-GPD hazard maps for hd{temperature}.")

# %%
