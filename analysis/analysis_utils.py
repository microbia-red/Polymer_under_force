# analysis_utils.py
#
# Reusable utility functions for the simulation analysis.
# Includes a robust Jackknife function and a safe CSV reader.

import os
import numpy as np
import pandas as pd
from pathlib import Path

def robust_read_csv(path: Path) -> pd.DataFrame | None:
    """
    Safely reads a CSV. Returns None if the file does not exist,
    is empty, or cannot be read.
    """
    if not path.is_file():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [Util Read Error] Could not read {path.name}: {e}")
        return None


def jackknife_analysis(data_input, n_bins: int, func, dict_input_type: str = 'single_dict'):
    """
    Performs a robust Jackknife analysis for a given function (func).
    
    Parameters:
    ----------
    data_input:
        Input data structure. Controlled by `dict_input_type`:
        
        1. 'single_dict' (default):
           A dict of 1D arrays, e.g.,
           {'ext': [1,2,3], 'Rg2': [4,5,6], 'energy': [7,8,9]}
           All arrays must have the same length.
           `func` will receive a sub-dictionary with this structure.

        2. 'dict_of_arrays':
           A dict of 1D arrays, e.g.,
           {F1: [E1, E2...], F2: [E3, E4...]}
           `func` will receive a sub-dictionary with this structure.

    n_bins: int
        Number of bins (blocks) for Jackknife resampling.

    func: callable
        The function that calculates the observable.
        Receives a dictionary (based on `dict_input_type`)
        and must return a single number or a numpy array.

    dict_input_type: str
        Options: 'single_dict', 'dict_of_arrays'.

    Returns:
    -------
    jk_mean: float or np.array
        The Jackknife mean of the observable.
        
    jk_err: float or np.array
        The Jackknife standard error of the observable.
    """
    
    jk_samples = [] # Stores results from each leave-one-out sample

    try:
        if dict_input_type == 'single_dict':
            # --- Case 1: Single dict of arrays (e.g., data from one T, F=0) ---
            N_data = len(next(iter(data_input.values())))
            n_B = N_data // n_bins # Points per bin
            if n_B == 0:
                raise ValueError(f"Not enough data ({N_data}) for {n_bins} bins.")
            
            use_len = n_B * n_bins
            # Slice data to an exact multiple of n_bins and create bins
            bins = {k: v[:use_len].reshape(n_bins, n_B) for k, v in data_input.items()}

            for i in range(n_bins):
                # Create the leave-one-out dataset (deleting bin 'i')
                reduced_data = {k: np.delete(bins[k], i, axis=0).reshape(-1) for k in bins}
                jk_samples.append(func(reduced_data))

        elif dict_input_type == 'dict_of_arrays':
            # --- Case 2: Dict of arrays (e.g., data from one T, all F) ---
            # Find the common minimum length
            min_len = min(len(v) for v in data_input.values())
            n_B = min_len // n_bins
            if n_B == 0:
                raise ValueError(f"Not enough data (min_len={min_len}) for {n_bins} bins.")
            
            use_len = n_B * n_bins
            # Truncate *all* arrays to this length and create bins
            bins = {key: val[:use_len].reshape(n_bins, n_B) 
                    for key, val in data_input.items()}

            for i in range(n_bins):
                # Create the leave-one-out dataset
                reduced_data = {key: np.delete(val_binned, i, axis=0).reshape(-1) 
                                for key, val_binned in bins.items()}
                jk_samples.append(func(reduced_data))

        else:
            raise ValueError(f"Unknown dict_input_type: {dict_input_type}")

    except ValueError as e:
        # Return NaN if Jackknife fails (e.g., insufficient data)
        print(f"  [Jackknife Warning] {e}")
        # Determine output shape of func with a small sample
        try:
            sample_output = func({k: v[:2] for k, v in data_input.items()} if dict_input_type == 'single_dict' else {k: v[:2] for k, v in data_input.items()})
            nan_array = np.full_like(sample_output, np.nan)
            return nan_array, nan_array
        except Exception:
             return np.nan, np.nan # Fallback for single value

    # Calculate mean and error from Jackknife samples
    jk_samples = np.array(jk_samples)
    jk_mean = jk_samples.mean(axis=0)
    
    # Jackknife variance formula
    # (n_bins-1)/n_bins * sum( (sample_i - mean)^2 )
    sum_sq_diff = np.sum((jk_samples - jk_mean)**2, axis=0)
    jk_var = (n_bins - 1) / n_bins * sum_sq_diff
    jk_err = np.sqrt(jk_var)
    
    return jk_mean, jk_err
