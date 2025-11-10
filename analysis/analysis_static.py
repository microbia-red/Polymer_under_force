# analysis_static.py
#
# Main script to analyze CONSTANT FORCE simulation results.
# Runs all static analyses (melting point, critical force)
# and generates output plots and CSV files.

import os
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# Import utility functions (like jackknife)
import analysis_utils as utils

# =============================================================================
# --- USER CONFIGURATION ---
# !! IMPORTANT: Edit these paths to match your folder structure
# Path to the results from Constant_force.cpp (containing T_... folders)
STATIC_RESULTS_PATH = Path(r"../results/Constant_force")
# Path where all plots and analysis CSVs will be saved
SAVE_PATH = Path("../analysis_results/static")
# =============================================================================

# --- Analysis Parameters ---
N_MONOMERS = 64
BURN_IN_SWEEPS = 100_000
N_BINS_JACKKNIFE = 50  # Number of bins for Jackknife analysis


def set_plot_style():
    """Sets a global Matplotlib style for consistency."""
    MAJ_TICK_LENGTH = 10
    MAJ_TICK_WIDTH = 1.5
    MIN_TICK_LENGTH = 5
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
            "text.usetex": True, "font.family": "serif", "font.size": 34,
        })
        plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
    except Exception:
        print("[Warning] LaTeX not available. Using standard font.")
        plt.rcParams.update({
            "text.usetex": False, "font.family": "sans-serif", "font.size": 24,
        })
    plt.rcParams['figure.figsize'] = [12, 7]
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['axes.grid'] = False
    print("[Info] Matplotlib style set.")


def analyze_melting_curve(base_path: Path, save_path: Path):
    """
    STEP 1: Calculates specific heat (Cv) and Rg susceptibility (d<Rg^2>/dT)
    as a function of Temperature, using only Zero Force (F=0) data.
    """
    print("\n--- Starting Static Analysis [1/2]: Melting Curve (F=0) ---")
    
    # {T: {'ext': array, 'Rg2': array, 'energy': array}}
    all_data_by_T = {} 

    # --- Load all F=0 data first ---
    f0_folders = sorted(glob.glob(str(base_path / "T_*" / "F_0.0000")))
    for folder in f0_folders:
        folder_path = Path(folder)
        try:
            T = float(folder_path.parent.name.split("_")[-1])
        except ValueError:
            continue

        f_ext = folder_path / "extension.csv"
        f_rg  = folder_path / "Rg.csv"
        f_en  = folder_path / "energy_noforce.csv"
        
        if not (f_ext.is_file() and f_rg.is_file() and f_en.is_file()):
            print(f"  Skipping T={T:.2f}: Missing required .csv files")
            continue

        df_ext = utils.robust_read_csv(f_ext)
        df_rg  = utils.robust_read_csv(f_rg)
        df_en  = utils.robust_read_csv(f_en)

        if any(df is None for df in [df_ext, df_rg, df_en]):
            continue

        df_ext = df_ext[df_ext["sweep"] >= BURN_IN_SWEEPS]
        df_rg  = df_rg[df_rg["sweep"] >= BURN_IN_SWEEPS]
        df_en  = df_en[df_en["sweep"] >= BURN_IN_SWEEPS]

        df = df_ext.merge(df_rg, on="sweep", how="inner").merge(df_en, on="sweep", how="inner")
        
        try:
            ext_col = [col for col in df.columns if "ext" in col][0]
            rg2_col = [col for col in df.columns if "Rg" in col][0]
            en_col  = [col for col in df.columns if "energy" in col][0]
        except IndexError:
            print(f"  Skipping T={T:.2f}: Could not find ext, Rg2, or energy columns.")
            continue

        if len(df) < (N_BINS_JACKKNIFE * 2): # Need at least 2 points per bin
            print(f"  Skipping T={T:.2f}: Not enough data after burn-in ({len(df)} points)")
            continue

        all_data_by_T[T] = {
            "ext": df[ext_col].values,
            "Rg2": df[rg2_col].values,
            "energy": df[en_col].values
        }

    if not all_data_by_T:
        print("[Error] No valid F=0 data found for analysis.")
        return

    print("  Calculating observables with Jackknife...")
    results_list = []
    
    for T, data_dict in all_data_by_T.items():
        beta = 1.0 / T

        # Observable 1: Specific Heat (Cv) per monomer
        def f_cv(d):
            E_mean = d["energy"].mean()
            E2_mean = (d["energy"]**2).mean()
            return (E2_mean - E_mean**2) / (N_MONOMERS * T**2)

        # Observable 2: Thermal derivative of Rg2 (d<Rg^2>/dT)
        def f_d_rg2_dt(d):
            # This is the covariance: d<A>/dT = beta^2 * ( <A*E> - <A><E> )
            Rg2_E_mean = (d["Rg2"] * d["energy"]).mean()
            Rg2_mean = d["Rg2"].mean()
            E_mean = d["energy"].mean()
            return (beta**2) * (Rg2_E_mean - Rg2_mean * E_mean)

        cv_val, cv_err = utils.jackknife_analysis(data_dict, N_BINS_JACKKNIFE, f_cv)
        drg2_val, drg2_err = utils.jackknife_analysis(data_dict, N_BINS_JACKKNIFE, f_d_rg2_dt)

        results_list.append({
            "T": T,
            "Cv_mean": cv_val, "Cv_err": cv_err,
            "dRg2_dT_mean": drg2_val, "dRg2_dT_err": drg2_err
        })

    df_results = pd.DataFrame(results_list).sort_values("T").reset_index(drop=True)
    csv_path = save_path / "static_analysis_F0_melting_curve.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  Results saved to: {csv_path.name}")

    # Plot Cv(T)
    plt.figure()
    plt.errorbar(df_results["T"], df_results["Cv_mean"], yerr=df_results["Cv_err"], 
                 fmt='o-', capsize=3, color='red', label=r'$C_v / N$')
    plt.xlabel("$T$")
    plt.ylabel(r"$C_v / N$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path / "plot_Cv_vs_T.png", dpi=300)
    plt.close()

    # Plot d<Rg^2>/dT
    plt.figure()
    plt.errorbar(df_results["T"], df_results["dRg2_dT_mean"], yerr=df_results["dRg2_dT_err"], 
                 fmt='s-', capsize=3, color='blue', label=r'$d\langle R_g^2 \rangle / dT$')
    plt.xlabel("$T$")
    plt.ylabel(r"$d\langle R_g^2 \rangle / dT$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path / "plot_dRg2_dT_vs_T.png", dpi=300)
    plt.close()
    print("  Melting curve plots generated.")


def analyze_critical_force(base_path: Path, save_path: Path):
    """
    STEP 2: Calculates Critical Force (Fc) vs. Temperature (T) and
    Max Specific Heat (C_max) vs. T.
    """
    print("\n--- Starting Static Analysis [2/2]: Phase Diagram (Fc vs T) ---")
    
    results = [] # List of {T, Fcrit, Fcrit_err, Cmax, Cmax_err}

    for T_folder in sorted(glob.glob(str(base_path / "T_*"))):
        try:
            T_value = float(os.path.basename(T_folder).split("_")[-1])
        except ValueError:
            continue

        print(f"  Processing T = {T_value:.2f}...")
        energy_data_by_F = {} # {F: array_of_energies}
        
        for F_folder in sorted(glob.glob(str(Path(T_folder) / "F_*"))):
            try:
                F_value = float(os.path.basename(F_folder).split("_")[-1])
            except ValueError:
                continue
            
            df = utils.robust_read_csv(Path(F_folder) / "energy_noforce.csv")
            if df is None: continue

            if "sweep" in df.columns:
                df = df[df["sweep"] >= BURN_IN_SWEEPS]
            if df.empty: continue

            col = [c for c in df.columns if c != "sweep"][0]
            E = pd.to_numeric(df[col], errors="coerce").dropna().values
            
            if len(E) >= (N_BINS_JACKKNIFE * 2):
                energy_data_by_F[F_value] = E
            else:
                print(f"    Skipping F={F_value:.4f}: Not enough data ({len(E)} points)")

        if not energy_data_by_F:
            print(f"    Skipping T={T_value:.2f}: No valid energy data found.")
            continue

        # Function to calc specific heat per monomer
        def f_cv(E, T):
            E_mean = E.mean()
            E2_mean = (E**2).mean()
            return (E2_mean - E_mean**2) / (N_MONOMERS * T**2)

        # Jackknife helper: receives {F: E_array} dict, returns (Fc, Cmax)
        def get_Fcrit_Cmax(E_dict):
            # 1. Calculate Cv for each Force
            cv_by_F = {F: f_cv(E, T_value) for F, E in E_dict.items()}
            # 2. Find the Force that maximizes Cv
            F_crit = max(cv_by_F, key=cv_by_F.get)
            C_max = cv_by_F[F_crit]
            return np.array([F_crit, C_max])

        (Fc_mean, Cmax_mean), (Fc_err, Cmax_err) = utils.jackknife_analysis(
            energy_data_by_F, 
            n_bins=N_BINS_JACKKNIFE, 
            func=get_Fcrit_Cmax,
            dict_input_type='dict_of_arrays'
        )
        
        if not np.isnan(Fc_mean):
            print(f"    T={T_value:.2f} -> Fc={Fc_mean:.4f} ± {Fc_err:.2e}")
            results.append({
                "T": T_value,
                "F_crit_mean": Fc_mean, "F_crit_err": Fc_err,
                "C_max_mean": Cmax_mean, "C_max_err": Cmax_err
            })

    if not results:
        print("[Error] No critical force results were generated.")
        return

    df_out = pd.DataFrame(results).sort_values("T").reset_index(drop=True)
    csv_path = save_path / "static_analysis_critical_force_vs_T.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path.name}")
    print(df_out.to_string(index=False, float_format="%.4f"))

    # Plot Fc(T)
    plt.figure()
    plt.errorbar(df_out["T"], df_out["F_crit_mean"], yerr=df_out["F_crit_err"], 
                 fmt='o-', capsize=3, color='blue')
    plt.xlabel("$T$")
    plt.ylabel("$f_c$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path / "plot_Fc_vs_T.png", dpi=300)
    plt.close()

    # Plot Cmax(T)
    plt.figure()
    plt.errorbar(df_out["T"], df_out["C_max_mean"], yerr=df_out["C_max_err"], 
                 fmt='s-', capsize=3, color='red')
    plt.xlabel("$T$")
    plt.ylabel(r"$C_{\max}/N$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path / "plot_Cmax_vs_T.png", dpi=300)
    plt.close()
    print("  Phase diagram plots generated.")


def main():
    """Main function to orchestrate the static analysis."""
    
    # Create save directory
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"[Info] Analysis results will be saved to: {SAVE_PATH.resolve()}")

    if not STATIC_RESULTS_PATH.is_dir():
        print(f"[ERROR] Base results path not found: {STATIC_RESULTS_PATH.resolve()}")
        print("Please edit 'STATIC_RESULTS_PATH' at the top of this script.")
        return

    set_plot_style()
    
    # --- STEP 1: F=0 Melting Curve ---
    analyze_melting_curve(STATIC_RESULTS_PATH, SAVE_PATH)
    
    # --- STEP 2: F>0 Phase Diagram ---
    analyze_critical_force(STATIC_RESULTS_PATH, SAVE_PATH)

    print("\n--- Static analysis complete. ---")

if __name__ == "__main__":
    main()
