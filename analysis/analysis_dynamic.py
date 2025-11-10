# analysis_dynamic.py
#
# Script to analyze PERIODIC FORCE simulation results.
# Plots the averaged hysteresis loops.

import os
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import utility functions
import analysis_utils as utils

# =============================================================================
# --- USER CONFIGURATION ---
# !! IMPORTANT: Edit these paths to match your folder structure
# Path to the results from Periodic_force.cpp (containing Fmax_... folders)
DYNAMIC_RESULTS_PATH = Path(r"../results/Periodic_force")
# Path where all plots and analysis CSVs will be saved
SAVE_PATH = Path("../analysis_results/dynamic")
# =============================================================================

# --- Analysis Parameters ---
# Example: plot loops for Fmax=5.0
FMAX_TO_PLOT = 5.0 
# Average over the first 100 cycles
N_CYCLES_TO_AVERAGE = 100


def set_plot_style():
    """Sets a global Matplotlib style for consistency."""
    MAJ_TICK_LENGTH = 20
    MAJ_TICK_WIDTH = 2
    MIN_TICK_LENGTH = 10
    MIN_TICK_WIDTH = 1
    mpl.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = MAJ_TICK_WIDTH
    plt.rcParams['xtick.major.size'] = MAJ_TICK_LENGTH
    plt.rcParams['xtick.minor.width'] = MIN_TICK_WIDTH
    plt.rcParams['xtick.minor.size'] = MIN_TICK_LENGTH
    plt.rcParams['ytick.major.width'] = MAJ_TICK_WIDTH
    plt.rcParams['ytick.major.size'] = MAJ_TICK_LENGTH
    plt.rcParams['ytick.minor.width'] = MIN_TICK_WIDTH
    plt.rcParams['ytick.minor.size'] = MIN_TICK_LENGTH
    try:
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif", "font.size": 44,
        })
        plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
    except Exception:
        print("[Warning] LaTeX not available. Using standard font.")
        plt.rcParams.update({
            "text.usetex": False, "font.family": "sans-serif", "font.size": 34,
        })
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['axes.grid'] = False
    print("[Info] Matplotlib style set.")


def plot_hysteresis_loops_with_error(base_dir: Path, save_dir: Path, Fmax: float, n_cycles: int):
    """Plots the average curve with error bars (std. error) across cycles."""
    print(f"\n--- Starting Dynamic Analysis: Hysteresis ---")
    print(f"Plotting averages for Fmax={Fmax}, over n_cycles={n_cycles}")
    
    pattern = str(base_dir / f"Fmax_{Fmax}_freq_*")
    freq_dirs = sorted(glob.glob(pattern))
    if not freq_dirs:
        print(f"[Error] No folders found for Fmax = {Fmax} in {base_dir}")
        return

    plt.figure()
    for freq_dir in freq_dirs:
        freq_dir_path = Path(freq_dir)
        freq_label = freq_dir_path.name.split("_freq_")[-1]
        cycle_dirs = sorted(glob.glob(str(freq_dir_path / "Cycle_*")))
        
        if n_cycles > 0:
            cycle_dirs = cycle_dirs[:n_cycles] # Limit to specified number
        
        if not cycle_dirs:
            print(f"  Skipping freq={freq_label}: No 'Cycle_*' data found.")
            continue

        loop_data = []
        for cdir in cycle_dirs:
            try:
                # Read only the 'f' and 'z' columns
                data = utils.robust_read_csv(Path(cdir) / "zeta.csv")
                if data is not None and 'f' in data.columns and 'z' in data.columns:
                    loop_data.append(data[['f','z']].values)
                else:
                    print(f"    [Warning] 'zeta.csv' in {cdir} is invalid.")
            except Exception as e:
                print(f"    [Error] Reading {cdir}: {e}")
                continue
        
        if not loop_data:
            print(f"  Skipping freq={freq_label}: No valid 'zeta.csv' data was read.")
            continue

        try:
            arr = np.stack(loop_data, axis=0)
        except ValueError as e:
            print(f"  Skipping freq={freq_label}: Mismatched data shapes. {e}")
            continue

        mean_f   = arr[0,:,0] # Assume 'f' is identical across all cycles
        zs       = arr[:,:,1] # Get all 'z' data
        mean_z   = zs.mean(axis=0)
        # Standard error of the mean
        err_z    = zs.std(axis=0, ddof=1) / np.sqrt(len(zs)) if len(zs) > 1 else np.zeros_like(mean_z)

        plt.errorbar(mean_f, mean_z, yerr=err_z,
                     label=f"$\\nu\\approx{freq_label}$",
                     capsize=3, elinewidth=1, marker='o', markersize=4)

    plt.xlabel("$f$")
    plt.ylabel(r"$\langle \Delta z \rangle$")
    plt.legend(loc="upper left", fontsize=35)
    plt.tight_layout()
    plt.subplots_adjust(left=0.17, bottom=0.2)

    output_path = save_dir / f"hysteresis_Fmax{Fmax}_avg_cycles_with_error.png"
    plt.savefig(output_path, dpi=300)
    print(f"  Plot saved to: {output_path.name}")
    plt.show()


def main():
    """Main function to orchestrate the dynamic analysis."""
    
    # Create save directory
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"[Info] Analysis results will be saved to: {SAVE_PATH.resolve()}")

    if not DYNAMIC_RESULTS_PATH.is_dir():
        print(f"[ERROR] Base results path not found: {DYNAMIC_RESULTS_PATH.resolve()}")
        print("Please edit 'DYNAMIC_RESULTS_PATH' at the top of this script.")
        return
        
    set_plot_style()
    
    plot_hysteresis_loops_with_error(
        DYNAMIC_RESULTS_PATH, 
        SAVE_PATH, 
        Fmax=FMAX_TO_PLOT, 
        n_cycles=N_CYCLES_TO_AVERAGE
    )
    
    print("\n--- Dynamic analysis complete. ---")

if __name__ == "__main__":
    main()
