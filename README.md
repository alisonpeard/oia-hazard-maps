
## d-GPD (for count data)

### Stationary (simplest) model

Steps:
- Choose 30-year windows around the epoch-of-interest
- Choose a threshold of zero (days of extreme temperatures)
- Fit a d-GPD to each 30-year window
- Calculate return levels

Limitations:
- Does not account for trend within the 30-year window
- Threshold of zero may be too low for GPD to be justified

### De-trended model

Steps:
- Fit a time-varying Poisson or Binomial GLM to the data
- Use it to subtract the time-varying expected value and obtain stationary residuals
- Choose a threshold of zero
- Fit a d-GPD to this
- Calculate return levels
- Re-add the time-varying expected values

Advantages:
- Can use all data for fits and extract more exceedances

Limitations:
- Threshold of zero may be inappropriate
- Doesn't account for multiplicative seasonality

### Non-stationary model

Steps:
- Same as detrended model
- Except time-varying scale parameter

Limitations:
- Threshold may be inappropriate

## References
- https://arxiv.org/abs/1707.05033