// A Monte Carlo simulation of a polymer under a periodic pulling force.

#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <gflags/gflags.h>

DEFINE_double(temperature,        1.0, "Set the temperature for the simulation");
DEFINE_double(max_force,          1.0, "Set the max. pulling force for the simulation");
DEFINE_double(delta_f,           0.10, "Increment of force between updates");
DEFINE_int32 (sweep_per_force,   1000, "Number of Monte Carlo sweeps per force step");
DEFINE_int32 (total_cycles,        10, "Number of complete force cycles");
DEFINE_int32 (seed,                 5, "RNG seed");
DEFINE_string(seed_file,    "./initial_positions.csv", "Path to CSV file with initial positions");
DEFINE_string(base_dir, "./results/Periodic_force", "Base directory for output files");

static constexpr int N = 64;

static constexpr double k_spring = 16.67;
static constexpr double B_lj = 1.0;
static constexpr double C_lj = 2.0;
static constexpr double d0 = 1.0;
static constexpr double cut_off = 2.5;

static constexpr int interval_default = 1;
static constexpr double target_acceptance = 0.40;
static constexpr double corr_coeff = 0.05;

static int GLOBAL_SEED = 5;

struct DataPoint {
  int sweep;
  double value;
};

struct ThreeDataPoint {
  double f;
  int sweep;
  double value;
};

struct SplitEnergyDataPoint {
  int sweep;
  double har;
  double lj;
};

struct AreaData {
  int cycle;
  double area;
};

static void flush_buffer_to_file(const std::vector<DataPoint>& buffer,
                                 std::ofstream &fout) {
  if (!fout) {
    std::cerr << "Error writing in the file" << std::endl;
    return;
  }
  for (const auto &dp : buffer) {
    fout << dp.sweep << ',' << dp.value << '\n';
  }
}

static void three_flush_buffer_to_file(const std::vector<ThreeDataPoint>& buffer,
                                       std::ofstream &fout) {
  if (!fout) {
    std::cerr << "Error writing in the file" << std::endl;
    return;
  }
  for (const auto &dp : buffer) {
    fout << dp.f << "," << dp.sweep << ',' << dp.value << '\n';
  }
}

static void split_energy_flush_buffer_to_file(const std::vector<SplitEnergyDataPoint>& buffer,
                                              std::ofstream &fout) {
  if (!fout) {
    std::cerr << "Error writing in the file" << std::endl;
    return;
  }
  for (const auto &dp : buffer) {
    fout << dp.sweep << ',' << dp.har << ',' << dp.lj << '\n';
  }
}

static void precompute_random_shifts(int total_proposals,
                                     std::vector<std::array<double,3>> &R) {
  R.clear();
  R.reserve(static_cast<size_t>(total_proposals));
  static std::mt19937 g(static_cast<unsigned>(GLOBAL_SEED));
  std::uniform_real_distribution<> dist(-0.5, 0.5);
  for (int i = 0; i < total_proposals; ++i) {
    double x = dist(g), y = dist(g), z = dist(g);
    double norm = std::sqrt(x*x + y*y + z*z);
    if (norm == 0.0) {
      R.push_back({1.0, 0.0, 0.0});
    } else {
      R.push_back({x / norm, y / norm, z / norm});
    }
  }
}

static void setup_files_and_directories_FORCE(double T, int cycle,
                                              const std::string &base,
                                              std::vector<std::ofstream> &fs,
                                              std::vector<std::string> &names) {
  std::ostringstream oss;
  int ms = static_cast<int>(std::ceil(FLAGS_max_force / FLAGS_delta_f));
  double frequency = 1.0 / (2.0 * static_cast<double>(ms) * static_cast<double>(FLAGS_sweep_per_force));
  oss << base << "/Fmax_" << FLAGS_max_force
      << "_freq_" << std::fixed << std::setprecision(7) << frequency
      << "/Cycle_" << cycle << "/";
  std::filesystem::create_directories(oss.str());
  names = {
    oss.str() + "extension.csv",
    oss.str() + "Rg.csv",
    oss.str() + "positions.csv",
    oss.str() + "energy_force.csv",
    oss.str() + "energy_noforce.csv",
    oss.str() + "split_energy.csv",
    oss.str() + "sm_acceptance.csv",
    oss.str() + "ts_acceptance.csv",
    oss.str() + "cutoff.txt",
    oss.str() + "hysteresis_area.csv",
    oss.str() + "zeta.csv"
  };
  fs.clear();
  for (auto &n : names) {
    fs.emplace_back(n);
  }
  // Tenemos 10 ficheros; escribe cabeceras
  if (fs.size() >= 10) {
    fs[0] << "f,sweep,ext\n";
    fs[1] << "f,sweep,Rg2\n";
    fs[2] << "x,y,z\n";
    fs[3] << "sweep,energy\n";
    fs[4] << "sweep,energy\n";
    fs[5] << "sweep,harmonic,LJ\n";
    fs[6] << "sweep,acc\n";
    fs[7] << "sweep,acc\n";
    fs[9] << "cycle,area\n";
    fs[10] << "f,sweep,z\n";
  }
}

static bool load_positions(const std::string &file, double pos[N][3]) {
  std::ifstream in(file);
  if (!in) return false;
  std::string line;
  std::vector<std::array<double,3>> v;
  while (std::getline(in, line) && v.size() < N) {
    if (line.empty()) continue;
    if (std::isalpha(static_cast<unsigned char>(line[0]))) continue;
    std::istringstream iss(line);
    double x, y, z; char c1, c2;
    if (!(iss >> x >> c1 >> y >> c2 >> z)) continue;
    if (c1 != ',' || c2 != ',') continue;
    v.push_back({x, y, z});
  }
  if (v.size() != N) return false;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 3; ++j) {
      pos[i][j] = v[i][j];
    }
  }
  return true;
}

static double end_to_end_length(const double positions[N][3]) {
  double dx = positions[N-1][0] - positions[0][0];
  double dy = positions[N-1][1] - positions[0][1];
  double dz = positions[N-1][2] - positions[0][2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static double end_to_end_z(const double positions[N][3]) {
  return std::abs(positions[N-1][2] - positions[0][2]);
}

static void centre_of_mass(const double positions[N][3], double cm[3]) {
  cm[0] = cm[1] = cm[2] = 0.0;
  for (int i = 0; i < N; ++i) {
    cm[0] += positions[i][0];
    cm[1] += positions[i][1];
    cm[2] += positions[i][2];
  }
  cm[0] /= static_cast<double>(N);
  cm[1] /= static_cast<double>(N);
  cm[2] /= static_cast<double>(N);
}

static double radius_of_gyration_squared(const double positions[N][3]) {
  double cm[3];
  centre_of_mass(positions, cm);
  double rog2 = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 3; ++j) {
      double diff = positions[i][j] - cm[j];
      rog2 += diff * diff;
    }
  }
  rog2 /= static_cast<double>(N);
  return rog2;
}

static double harmonic_potential(const double positions[N][3]) {
  double hp = 0.0;
  for (int i = 0; i < N - 1; ++i) {
    double dx = positions[i+1][0] - positions[i][0];
    double dy = positions[i+1][1] - positions[i][1];
    double dz = positions[i+1][2] - positions[i][2];
    double d = std::sqrt(dx*dx + dy*dy + dz*dz);
    double deviation = d - d0;
    hp += 0.5 * k_spring * deviation * deviation;
  }
  return hp;
}

static double lennard_jones(const double positions[N][3]) {
  double lj = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = i + 2; j < N; ++j) {
      double dx = positions[j][0] - positions[i][0];
      double dy = positions[j][1] - positions[i][1];
      double dz = positions[j][2] - positions[i][2];
      double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (d < cut_off) {
        double d6  = d*d*d*d*d*d;
        double d12 = d6 * d6;
        lj += (B_lj / d12) - (C_lj / d6);
      }
    }
  }
  return lj;
}

static void initial_state_matrix(const double positions[N][3],
                                 double energy[N][N],
                                 double sums[N],
                                 double &total_energy) {
  for (int i = 0; i < N; ++i) {
    sums[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      energy[i][j] = 0.0;
    }
  }
  total_energy = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double e = 0.0;
      if (j == i + 1) {
        double dx = positions[j][0] - positions[i][0];
        double dy = positions[j][1] - positions[i][1];
        double dz = positions[j][2] - positions[i][2];
        double d = std::sqrt(dx*dx + dy*dy + dz*dz);
        double deviation = d - d0;
        e = 0.5 * k_spring * deviation * deviation;
      } else {
        double dx = positions[j][0] - positions[i][0];
        double dy = positions[j][1] - positions[i][1];
        double dz = positions[j][2] - positions[i][2];
        double d = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d < cut_off) {
          double d6  = d*d*d*d*d*d;
          double d12 = d6 * d6;
          e = (B_lj / d12) - (C_lj / d6);
        }
      }
      energy[i][j] = energy[j][i] = e;
      sums[i] += e;
      sums[j] += e;
      total_energy += e;
    }
  }
}

static void sm_update(const double positions[N][3],
                      double energy[N][N],
                      double sums[N],
                      double &total_energy,
                      int idx) {
  //double old_sum = sums[idx];
  double new_sum = 0.0;
  double delta_total = 0.0;
  for (int i = 0; i < N; ++i) {
    if (i == idx) continue;
    double new_e = 0.0;
    if (std::abs(i - idx) == 1) {
      double dx = positions[i][0] - positions[idx][0];
      double dy = positions[i][1] - positions[idx][1];
      double dz = positions[i][2] - positions[idx][2];
      double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      double deviation = d - d0;
      new_e = 0.5 * k_spring * deviation * deviation;
    } else {
      double dx = positions[i][0] - positions[idx][0];
      double dy = positions[i][1] - positions[idx][1];
      double dz = positions[i][2] - positions[idx][2];
      double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (d < cut_off) {
        double d6  = d*d*d*d*d*d;
        double d12 = d6 * d6;
        new_e = (B_lj / d12) - (C_lj / d6);
      }
    }
    double old_e = energy[i][idx];
    double delta = new_e - old_e;
    energy[i][idx] = new_e;
    energy[idx][i] = new_e;
    new_sum += new_e;
    sums[i] += delta;
    delta_total += delta;
  }
  sums[idx] = new_sum;
  total_energy += delta_total;
}

static void ts_update(const double positions[N][3],
                      double energy[N][N],
                      double sums[N],
                      double &total_energy,
                      int index) {
  for (int i = index; i < N; i++) {
    sm_update(positions,energy,sums,total_energy,i);
  }
}

static void montecarlo(double positions[N][3], double beta,
                       std::ofstream &f_ext,
                       std::ofstream &f_rog,
                       std::ofstream &f_pos,
                       std::ofstream &f_en_f,
                       std::ofstream &f_en_nf,
                       std::ofstream &f_split,
                       std::ofstream &f_sm_acc,
                       std::ofstream &f_ts_acc,
                       std::ofstream &f_co,
                       std::ofstream &f_area,
                       std::ofstream &f_zeta,
                       int cycle) {
  std::vector<DataPoint> buf_en_f, buf_en_nf;
  std::vector<DataPoint> buf_sm_acc, buf_ts_acc;
  std::vector<ThreeDataPoint> buf_ext, buf_rog2, buf_zeta;
  std::vector<SplitEnergyDataPoint> buf_split;
  double newpositions[N][3];
  std::memcpy(newpositions, positions, sizeof(newpositions));
  int interval = interval_default;
  double sm_step_size = 0.19;
  double sm_proposed_moves = 0.0;
  double sm_accepted_moves = 0.0;
  double ts_step_size = 0.06;
  double ts_proposed_moves = 0.0;
  double ts_accepted_moves = 0.0;
  double energy[N][N];
  double newenergy[N][N];
  double sums[N];
  double new_sums[N];
  double total_energy = 0.0;
  double new_total_energy = 0.0;
  initial_state_matrix(positions, energy, sums, total_energy);
  for (int i_ = 0; i_ < N; ++i_) {
    new_sums[i_] = sums[i_];
    for (int j_ = 0; j_ < N; ++j_) {
      newenergy[i_][j_] = energy[i_][j_];
    }
  }
  new_total_energy = total_energy;
  int ms = static_cast<int>(std::ceil(FLAGS_max_force / FLAGS_delta_f));
  int total_steps = 2 * ms;
  int total_proposals = total_steps * FLAGS_sweep_per_force * N + 1;
  std::vector<std::array<double,3>> shifts;
  precompute_random_shifts(total_proposals, shifts);

  std::mt19937 gen(static_cast<unsigned>(GLOBAL_SEED));
  std::uniform_int_distribution<> uid(1, N - 1);
  std::uniform_real_distribution<> ud(0.0, 1.0);
  std::uniform_real_distribution<> udts(0.0, 1.0);

  int propose = 0;
  double f_current = 0.0;
  std::vector<double> ext_up(ms + 1, 0.0);
  std::vector<double> ext_down(ms + 1, 0.0);
  for (int step_force = 0; step_force < ms; ++step_force) {
    f_current = step_force * FLAGS_delta_f;
    for (int sweep = 0; sweep < FLAGS_sweep_per_force; sweep++) {
      for (int step = 0; step < N; ++step) {
        int idx = uid(gen);
        const auto &sh = shifts[propose++];
        if (udts(gen) < 0.85) {
          sm_proposed_moves += 1.0;
          newpositions[idx][0] += sh[0] * sm_step_size;
          newpositions[idx][1] += sh[1] * sm_step_size;
          newpositions[idx][2] += sh[2] * sm_step_size;
          double Eold = total_energy - (f_current * end_to_end_z(positions));
          sm_update(newpositions, newenergy, new_sums, new_total_energy, idx);
          double Enew = new_total_energy - (f_current * end_to_end_z(newpositions));
          double dE = Enew - Eold;
          bool accept = false;
          if (dE < 0.0 || ud(gen) < std::exp(-dE * beta)) {
            accept = true;
          }
          if (accept) {
            sm_accepted_moves += 1.0;
            positions[idx][0] = newpositions[idx][0];
            positions[idx][1] = newpositions[idx][1];
            positions[idx][2] = newpositions[idx][2];
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                energy[ii][jj] = newenergy[ii][jj];
              }
              sums[ii] = new_sums[ii];
            }
            total_energy = new_total_energy;
          } else {
            newpositions[idx][0] = positions[idx][0];
            newpositions[idx][1] = positions[idx][1];
            newpositions[idx][2] = positions[idx][2];
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                newenergy[ii][jj] = energy[ii][jj];
              }
              new_sums[ii] = sums[ii];
            }
            new_total_energy = total_energy;
          }
        } else {
          ts_proposed_moves += 1.0;
          for (int ts = idx; ts < N; ts++) {
            newpositions[ts][0] += sh[0] * ts_step_size;
            newpositions[ts][1] += sh[1] * ts_step_size;
            newpositions[ts][2] += sh[2] * ts_step_size;
          }
          double Eold = total_energy - (f_current * end_to_end_z(positions));
          ts_update(newpositions, newenergy, new_sums, new_total_energy, idx);
          double Enew = new_total_energy - (f_current * end_to_end_z(newpositions));
          double dE = Enew - Eold;
          bool accept = false;
          if (dE < 0.0 || udts(gen) < std::exp(-dE * beta)) {
            accept = true;
          }
          if (accept) {
            ts_accepted_moves += 1.0;
            for (int k1 = idx; k1 < N; k1++) {
              positions[k1][0] = newpositions[k1][0];
              positions[k1][1] = newpositions[k1][1];
              positions[k1][2] = newpositions[k1][2];
            }
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                energy[ii][jj] = newenergy[ii][jj];
              }
              sums[ii] = new_sums[ii];
            }
            total_energy = new_total_energy;
          } else {
            for (int k2 = idx; k2 < N; k2++) {
              newpositions[k2][0] = positions[k2][0];
              newpositions[k2][1] = positions[k2][1];
              newpositions[k2][2] = positions[k2][2];
            }
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                newenergy[ii][jj] = energy[ii][jj];
              }
              new_sums[ii] = sums[ii];
            }
            new_total_energy = total_energy;
          }
        }
      }
      if (sweep % interval == 0) {
        double sm_p = (sm_accepted_moves / (sm_proposed_moves + 1e-15));
        double ts_p = (ts_accepted_moves / (ts_proposed_moves + 1e-15));
        buf_sm_acc.push_back({sweep, sm_p});
        buf_ts_acc.push_back({sweep, ts_p});
        buf_ext.push_back({f_current, sweep, end_to_end_length(positions)});
        buf_rog2.push_back({f_current, sweep, radius_of_gyration_squared(positions)});
        buf_zeta.push_back({f_current, sweep, end_to_end_z(positions)});
        buf_en_nf.push_back({sweep, total_energy});
        buf_en_f.push_back({sweep, total_energy - (f_current * end_to_end_z(positions))});
        buf_split.push_back({sweep, harmonic_potential(positions), lennard_jones(positions)});
      }
    }
     ext_up[step_force] = end_to_end_z(positions);
  }
  for (int step_force = ms; step_force >= 0; --step_force) {
    f_current = step_force * FLAGS_delta_f;
    for (int sweep = 0; sweep < FLAGS_sweep_per_force; sweep++) {
      for (int step = 0; step < N; ++step) {
        int idx = uid(gen);
        const auto &sh = shifts[propose++];
        if (udts(gen) < 0.85) {
          sm_proposed_moves += 1.0;
          newpositions[idx][0] += sh[0] * sm_step_size;
          newpositions[idx][1] += sh[1] * sm_step_size;
          newpositions[idx][2] += sh[2] * sm_step_size;
          double Eold = total_energy - (f_current * end_to_end_z(positions));
          sm_update(newpositions, newenergy, new_sums, new_total_energy, idx);
          double Enew = new_total_energy - (f_current * end_to_end_z(newpositions));
          double dE = Enew - Eold;
          bool accept = false;
          if (dE < 0.0 || ud(gen) < std::exp(-dE * beta)) {
            accept = true;
          }
          if (accept) {
            sm_accepted_moves += 1.0;
            positions[idx][0] = newpositions[idx][0];
            positions[idx][1] = newpositions[idx][1];
            positions[idx][2] = newpositions[idx][2];
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                energy[ii][jj] = newenergy[ii][jj];
              }
              sums[ii] = new_sums[ii];
            }
            total_energy = new_total_energy;
          } else {
            newpositions[idx][0] = positions[idx][0];
            newpositions[idx][1] = positions[idx][1];
            newpositions[idx][2] = positions[idx][2];
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                newenergy[ii][jj] = energy[ii][jj];
              }
              new_sums[ii] = sums[ii];
            }
            new_total_energy = total_energy;
          }
        } else {
          ts_proposed_moves += 1.0;
          for (int ts = idx; ts < N; ts++) {
            newpositions[ts][0] += sh[0] * ts_step_size;
            newpositions[ts][1] += sh[1] * ts_step_size;
            newpositions[ts][2] += sh[2] * ts_step_size;
          }
          double Eold = total_energy - (f_current * end_to_end_z(positions));
          ts_update(newpositions, newenergy, new_sums, new_total_energy, idx);
          double Enew = new_total_energy - (f_current * end_to_end_z(newpositions));
          double dE = Enew - Eold;
          bool accept = false;
          if (dE < 0.0 || udts(gen) < std::exp(-dE * beta)) {
            accept = true;
          }
          if (accept) {
            ts_accepted_moves += 1.0;
            for (int k1 = idx; k1 < N; k1++) {
              positions[k1][0] = newpositions[k1][0];
              positions[k1][1] = newpositions[k1][1];
              positions[k1][2] = newpositions[k1][2];
            }
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                energy[ii][jj] = newenergy[ii][jj];
              }
              sums[ii] = new_sums[ii];
            }
            total_energy = new_total_energy;
          } else {
            for (int k2 = idx; k2 < N; k2++) {
              newpositions[k2][0] = positions[k2][0];
              newpositions[k2][1] = positions[k2][1];
              newpositions[k2][2] = positions[k2][2];
            }
            for (int ii = 0; ii < N; ++ii) {
              for (int jj = 0; jj < N; ++jj) {
                newenergy[ii][jj] = energy[ii][jj];
              }
              new_sums[ii] = sums[ii];
            }
            new_total_energy = total_energy;
          }
        }
      }
      if (sweep % interval == 0) {
        double sm_p = (sm_accepted_moves / (sm_proposed_moves + 1e-15));
        double ts_p = (ts_accepted_moves / (ts_proposed_moves + 1e-15));
        buf_sm_acc.push_back({sweep, sm_p});
        buf_ts_acc.push_back({sweep, ts_p});
        buf_ext.push_back({f_current, sweep, end_to_end_length(positions)});
        buf_rog2.push_back({f_current, sweep, radius_of_gyration_squared(positions)});
        buf_zeta.push_back({f_current, sweep, end_to_end_z(positions)});
        buf_en_nf.push_back({sweep, total_energy});
        buf_en_f.push_back({sweep, total_energy - (f_current * end_to_end_z(positions))});
        buf_split.push_back({sweep, harmonic_potential(positions), lennard_jones(positions)});
      }
    }
    ext_down[step_force] = end_to_end_z(positions);
  }
  three_flush_buffer_to_file(buf_ext, f_ext);
  three_flush_buffer_to_file(buf_rog2, f_rog);
  three_flush_buffer_to_file(buf_zeta, f_zeta);
  flush_buffer_to_file(buf_en_f, f_en_f);
  flush_buffer_to_file(buf_en_nf, f_en_nf);
  split_energy_flush_buffer_to_file(buf_split, f_split);
  flush_buffer_to_file(buf_sm_acc, f_sm_acc);
  flush_buffer_to_file(buf_ts_acc, f_ts_acc);
  for (int i = 0; i < N; ++i) {
    f_pos << positions[i][0] << ',' << positions[i][1] << ',' << positions[i][2] << '\n';
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double dx = positions[i][0] - positions[j][0];
      double dy = positions[i][1] - positions[j][1];
      double dz = positions[i][2] - positions[j][2];
      double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (d < cut_off) {
        f_co << "1 ";
      } else {
        f_co << "0 ";
      }
    }
    f_co << '\n';
  }
  double area = 0.0;
  for (int i = 0; i < ms; ++i) {
    double y_up_avg = 0.5 * (ext_up[i] + ext_up[i+1]);
    double y_down_avg = 0.5 * (ext_down[i] + ext_down[i+1]);
    area += (y_up_avg - y_down_avg) * FLAGS_delta_f;
  }
  area = std::fabs(area);
  f_area << cycle << "," << area << "\n";
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  double T = FLAGS_temperature;
  int cycles = FLAGS_total_cycles;
  GLOBAL_SEED = FLAGS_seed;
  std::string base_dir = FLAGS_base_dir;
  std::string seed_file = FLAGS_seed_file;
  
  if (seed_file.empty()) {
    std::cerr << "Error: --seed_file must be provided and point to a CSV file" << std::endl;
    return 1;
  }
  double beta = 1.0 / T;
  double positions[N][3];
  if (!load_positions(seed_file, positions)) {
    std::cerr << "Error: failed to load seed positions from " << seed_file << std::endl;
    return 1;
  }
  for (int cycle = 0; cycle < cycles; cycle++) {
    std::cout << "Cycle " << cycle << " of " << cycles << " - F.max = " << FLAGS_max_force << std::fixed << std::setprecision(2) << std::endl;
    std::vector<std::ofstream> fs;
    std::vector<std::string> names;
    setup_files_and_directories_FORCE(T, cycle, base_dir, fs, names);
    montecarlo(positions, beta,
               fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7], fs[8], fs[9], fs[10], cycle);
    for (auto &o : fs) o.close();
  }
  std::cout << "End." << std::endl;
  return 0;
}
