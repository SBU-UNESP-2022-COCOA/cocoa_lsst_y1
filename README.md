# cocoa_lsst_y1

This branch is dedicated to the Growth-Geometry split (see: https://arxiv.org/abs/2010.05924, https://arxiv.org/abs/1410.5832) test in simulated lsst-y1 analysis.

In order to calculate the non-linear matter power spectrum , we are going to load the [Bacco Emulator](https://baccoemu.readthedocs.io/en/latest/). This requires some extra packages not included in the original [Cocoa](https://github.com/SBU-UNESP-2022-COCOA/cocoa2) installation.

Similar to the original Cocoa repository, the more straightforward way to install most prerequisites is via Conda. Cocoa's internal scripts will then install any remaining missing packages, using a provided internal cache located at cocoa_installation_libraries. Assuming that the user had previously installed Minicoda (or Anaconda), the first step is to type the following commands to create the cocoa CondaEmu environment.

conda create --name cocoaemu python=3.7 --quiet --yes && \
conda install -n cocoaemu --quiet --yes  \
  'conda-forge::libgcc-ng=10.3.0' \
  'conda-forge::libstdcxx-ng=10.3.0' \
  'conda-forge::libgfortran-ng=10.3.0' \
  'conda-forge::gxx_linux-64=10.3.0' \
  'conda-forge::gcc_linux-64=10.3.0' \
  'conda-forge::gfortran_linux-64=10.3.0' \
  'conda-forge::openmpi=4.1.1' \
  'conda-forge::sysroot_linux-64=2.17' \
  'conda-forge::git=2.33.1' \
  'conda-forge::git-lfs=3.0.2' \
  'conda-forge::hdf5=1.10.6' \
  'conda-forge::git-lfs=3.0.2' \
  'conda-forge::cmake=3.21.3' \
  'conda-forge::boost=1.76.0' \
  'conda-forge::gsl=2.7' \
  'conda-forge::fftw=3.3.10' \
  'conda-forge::cfitsio=4.0.0' \
  'conda-forge::openblas=0.3.18' \
  'conda-forge::lapack=3.9.0' \
  'conda-forge::armadillo=10.7.3'\
  'conda-forge::expat=2.4.1' \
  'conda-forge::cython=0.29.24' \
  'conda-forge::numpy=1.21.4' \
  'conda-forge::scipy=1.7.2' \
  'conda-forge::pandas=1.3.4' \
  'conda-forge::mpi4py=3.1.2' \
  'conda-forge::matplotlib=3.5.0' \
  'conda-forge::astropy=4.3.1' \
  'conda-forge::tbb-devel=2021.5.0' \
  'conda-forge::progressbar2=3.38.0' \
  'conda-forge::scikit-learn=1.0.2' \
  'conda-forge::keras=2.3.1' \
  'conda-forge::tensorflow=2.7.0'
  
  
  
