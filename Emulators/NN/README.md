# COLA Neural Network Emulator 1
## Author - João Victor S. Rebouças, October 2022

This is a nonlinear boost emulator for $\Lambda$CDM. It uses data from 400 COLA simulations in a Latin Hypercube Sample over the parameter space $\Omega_m, \Omega_b, n_s, A_s, h$, processed into $\log (B)$.

Usage:
 1. Clone the repository. The repository contains necessary data to perform the emulation: log boosts from simulations and reference cosmology spectra. It also contains the training code, a test notebook, an auxiliary library and the emulator code.
 2. Import the emulator code `colaemulator1.py` into your project.
 3. The usage is the same as Euclid Emulator 2. It is based on a function `get_boost` with same call signature as EE2.

