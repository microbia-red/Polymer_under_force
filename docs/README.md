# Polymer Simulations under External Force

This project contains C++ and Python implementations for Monte Carlo simulations of a **self-interacting polymer** under different external forces:
- **Constant force** (`Constant_force.cpp`)
- **Periodic force** (`Periodic_force.cpp`)

It was developed as part of my **Bachelor's thesis** at Universität Leipzig, supervised by M. Sc. Dustin Warkotsch.

---

## 🔬 Motivation
Mechanical forces play a crucial role in biological processes such as DNA unzipping, protein unfolding, and polymer collapse.  
Here, I simulate the response of a self-interacting polymer to external forces and study:
- Force–extension curves
- Hysteresis under periodic driving
- Phase transitions (coil–globule, stretched states)

The models are inspired by:
- Mishra et al., *Scaling of hysteresis loop of interacting polymers* (J. Chem. Phys. 138, 244905, 2013) 【https://doi.org/10.1063/1.4809985】

---

## 📂 Repository structure

```
/src
  Constant_force.cpp      # Polymer under constant pulling force
  Periodic_force.cpp      # Polymer under periodic driving
/analysis                 # Python scripts for data analysis
  analysis_static.py      # (Fc vs T, Cv vs T)
  analysis_dynamic.py     # (Hysteresis loops)
  analysis_utils.py       # (Helper functions, Jackknife)
/data
  example_input.csv       # Initial monomer positions (example)
/results
  figures/                # Example plots (hysteresis, force-extension, etc.)
/docs
  thesis_summary.pdf      # Short summary of the thesis
  README.md
  LICENSE
```

---

## ⚙️ Installation & Usage

### Requirements
- **C++17** (tested with g++)
- **Python 3.9+** (for analysis and plotting)
- Libraries: `numpy`, `pandas`, `matplotlib`

### Compile

# Example compilation for Constant_force.cpp
g++ -Wall -O2 -march=native src/Constant_force.cpp -lgflags -o constant_force

# Example compilation for Periodic_force.cpp
g++ -Wall -O2 -march=native src/Periodic_force.cpp -lgflags -o periodic_force

### Run

# Run a single simulation
./constant_force --temperature 1.0 --force 0.5 --sweep 500000 --seed_file data/example_input.csv --base_dir ./results/Constant_force

# Run multiple simulations in parallel (example using GNU Parallel)
parallel -j 10 --ungroup ./constant_force --temperature={1} --force={2} --base_dir=./results/Constant_force ::: $(echo 0.5 1 1.5 2) ::: $(echo 0.1 0.2 0.3 0.4)

---

### Analysis
The analysis code is located in the /analysis folder.

Before you run: You must first edit the configuration paths at the top of analysis_static.py and analysis_dynamic.py. Change the BASE_PATH variables to point to the directory where your simulation results are stored.

# example in analysis_static.py
STATIC_RESULTS_PATH = Path(r"./results/Constant_force")
# example in analysis_dynamic.py
DYNAMIC_RESULTS_PATH = Path(r"./results/Periodic_force")

## 📊 Results
Some of the reproduced results include:
- Force–extension curves at different temperatures
- Hysteresis loops under periodic force
- Scaling of loop area with frequency and amplitude

---

## 📜 License
This project is released under the [GNU GPLv3 License](LICENSE).[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## 👤 Author
- **J. Alberto S.P.**  
  - Thesis: *Self interacting polymer under a periodic force.*  
  - Supervisors: Prof. Dr. Wolfhard Janke, M. Sc. Dustin Warkotsch 
  - Contact: josea.spastor@gmail.com

---

## 🌐 References
1. Mishra, R. K. et al. *Scaling of hysteresis loop of interacting polymers under a periodic force.* J. Chem. Phys. 138, 244905 (2013). [https://doi.org/10.1063/1.4809985]



