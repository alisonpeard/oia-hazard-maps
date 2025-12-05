# %%
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pprint import pprint

VAR = "hd35"
THRESH = 'q80'

# walk the directories from 
datadir = os.path.join("..", "data", "hazard", VAR, "ensemble", THRESH)
SCENS = os.listdir(datadir);print(SCENS)
SCENS = [d for d in SCENS if not d.startswith(".")]
models = {}
for scen in SCENS:
    model_dirs = os.listdir(os.path.join(datadir, scen))
    models[scen] = model_dirs
    print(f"Scenario: {scen}, Models: {len(model_dirs)}")

pprint(models)

# lets find the models that are common to all scenarios
common_models = set(models[SCENS[0]])
for scen in ["rcp26", "rcp45", "rcp85"]:
    common_models = common_models.intersection(set(models[scen]))
common_models = sorted(list(common_models))
for cm in common_models: print(f"Common model: {cm}")
# %%
SCENS = ["rcp26", "rcp45", "rcp85"]
RP = 100
EPOCH = 2050

for MODEL in common_models:

    ds_scen = {}
    vmin = float("inf")
    vmax = -float("inf")
    for SCEN in SCENS:
        wd = os.path.join(
            "..", "data", "hazard", VAR, "ensemble", THRESH, SCEN
        )
        model_path = os.path.join(wd, MODEL, "return_levels.nc")
        ds = xr.open_dataset(model_path, engine="netcdf4", decode_times=False)
        # set datatype for all variables to float32 to save memory
        ds = ds.astype("float32")

        ds_scen[SCEN] = ds.sel(epoch=EPOCH, return_period=RP)
        vmin = min(vmin, ds_scen[SCEN][VAR].min(skipna=True).item())
        vmax = max(vmax, ds_scen[SCEN][VAR].max(skipna=True).item())

    fig, axs = plt.subplots(
        ncols=len(SCENS),
        nrows=1,
        figsize=(12, 3),
        constrained_layout=True,
        )

    for ax, SCEN in zip(axs.flat, SCENS):
        ds_scen[SCEN][VAR].plot(
            ax=ax, vmin=0, vmax=360, cmap="YlOrRd"
        )
    fig.suptitle(f"Model: {MODEL}, Epoch: {EPOCH}, RP: {RP} years")


# %%
