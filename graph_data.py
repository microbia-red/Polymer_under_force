import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 44,
})
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['axes.grid'] = False

def graph_data():

    base_dir = f"./results/Constant_force/T_1.00/F_0.5000"

    file_1 = pd.read_csv(os.path.join(base_dir, "extension.csv"))
    file_2 = pd.read_csv(os.path.join(base_dir, "Rg.csv"))
    file_3 = pd.read_csv(os.path.join(base_dir, "energy_noforce.csv"))
    file_4 = pd.read_csv(os.path.join(base_dir, "sm_step_size.csv"))
    file_5 = pd.read_csv(os.path.join(base_dir, "ts_step_size.csv"))
    file_6 = pd.read_csv(os.path.join(base_dir, "sm_acceptance.csv"))
    file_7 = pd.read_csv(os.path.join(base_dir, "ts_acceptance.csv"))

    
    plt.figure()
    plt.plot(file_1['sweep'], file_1['ext'], color='b')
    plt.xlabel('Sweeps')
    plt.ylabel('Extension')
    plt.title('Sweeps vs Extension')
    plt.grid(True)
    plt.xlim(400000,500000)

    plt.figure()
    plt.plot(file_2['sweep'], file_2['Rg2'], color='g')
    plt.xlabel('Sweeps')
    plt.ylabel('Radius of gyration squared')
    plt.title('Sweeps vs Radius of gyration squared')
    plt.grid(True)

    plt.figure()
    plt.plot(file_3.loc[mask, 'sweep'], file_3.loc[mask, 'energy'], color='r')
    plt.xlabel('Sweeps')
    plt.ylabel(r'$\langle E \rangle$')
    plt.title('Sweeps vs Energy')
    plt.grid(True)

    plt.figure()
    plt.plot(file_4['sweep'], file_4['ss'], color='b')
    plt.xlabel('Sweeps')
    plt.ylabel('Step size')
    plt.title('Sweeps vs Single monomer step size')
    plt.grid(True)

    plt.figure()
    plt.plot(file_5['sweep'], file_5['ss'], color='g')
    plt.xlabel('Sweeps')
    plt.ylabel('Step size')
    plt.title('Sweeps vs Tail shift step size')
    plt.grid(True)

    plt.figure()
    plt.plot(file_6['sweep'], file_6['acc'], color='r')
    plt.xlabel('Sweeps')
    plt.ylabel('Acceptance rate')
    plt.title('Sweeps vs Single monomer acc. rate')
    plt.grid(True)

    plt.figure()
    plt.plot(file_7['sweep'], file_7['acc'], color='r')
    plt.xlabel('Sweeps')
    plt.ylabel('Acceptance rate')
    plt.title('Sweeps vs Tail shift acc. rate')
    plt.grid(True)
    plt.show()

graph_data()
