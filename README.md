# Conical-Billiards
Code for manuscript "Transition to chaos with conical billiards" 

This repository contains Python functions for simulating ray trajectories, bouncing dynamics, and angular mappings on conical or elliptical geometries. The goal is to study particle or light-like paths under curvature, periodic boundary conditions using a variety of numerical techniques.

---

## Features

- Simulate ray trajectories on conical surfaces 
- Model elastic bounces off curved elliptical/conical boundaries
- Compute Poincaré maps and visualize phase space dynamics
- Implement custom boundary conditions and path evolution

---

## Installation

Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib scipy
```

This module requires only standard scientific libraries and does not depend on any external datasets.

---

## Usage

```python
from PoincareMapFunctions import *

# Example: Simulate path on conical surface with initial angle and angular defect

flist, rlist, crosslist, boundlist, direcfirstbounce=makepathelipseWChifirstbounce(gamma, chi,theta, a, h=h, R = R, dt = dt, numsteps=numsteps, direc = 'right', numbounce=numbounce , bound= bound, boundprime = boundprime)


# Plot the result
xlist = rlist*np.cos(flist)
ylist = rlist*np.sin(flist)
listit = makelistit(chi, gamma,flist, rlist)
plt.scatter(xlist, ylist, c= 'purple', s = 1)
```

---

## Key Functions

| `makepathelipseWChi(...)` | Simulates ray trajectory on a conical sector with elliptical boundary |

| `fvtAtoTheta(...)` | Determines poincare map for range of initial conditions (alist and phi values) |

| 'distribution(...)'| Determines distribution of random chords on cone |

| 'findexplist(...)'| Finds list of fractal dimension as number of timesteps increases |


## Outputs

Typical outputs include:

- Trajectories in polar and Cartesian coordinates
- Boundary interaction indices
- Plots of Poincaré sections and angle-angle maps

All data can be saved via `run_simulation(...)` into CSV files using informative filenames.

---

## Theory

This code models geodesic-like trajectories on conical spaces with angular cuts. It is suitable for exploring:

- Poincaré maps
- Hyperbolic or elliptic boundary reflections
- Effects of curvature and global topology on particle dynamics
---

## File Overview

- `PoincareMapFunctions.py` — Core simulation and analysis functions
- ' Functions used for Figures 7,17,18' — Additional functions for creating particular figures in manuscript
- `README.md` — You are here
- `*.csv` — Output data (e.g., `flist_rlist.csv`, `crosslist.csv`)

---

## License

This project is provided as-is for academic or research use. No warranty is provided.

---

## Author

Created by Lara Braverman
Feel free to reach out for collaboration or questions.
