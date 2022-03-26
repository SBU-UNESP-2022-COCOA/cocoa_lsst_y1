# Growth-Geometry split in LSST-Y1 analysis

This branch is dedicated to implementation of the Growth-Geometry split (see: https://arxiv.org/abs/2010.05924 and https://arxiv.org/abs/1410.5832) in simulated lsst-y1 analysis.

To calculate the non-linear matter power spectrum, we will load the [Bacco](https://bacco.dipc.org/emulator.html) and the [Euclid](https://github.com/miknab/EuclidEmulator2) Emulators. These emulators require some extra packages not included in the original [Cocoa](https://github.com/SBU-UNESP-2022-COCOA/cocoa2) installation.

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
        'conda-forge::hdf5=1.12.1' \
        'conda-forge::git-lfs=3.0.2' \
        'conda-forge::cmake=3.21.3' \
        'conda-forge::boost=1.77.0' \
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
        'conda-forge::six=1.13.0' \
        'conda-forge::typing_extensions=3.7.4.1' \
        'conda-forge::libgomp=10.3.0' \
        'conda-forge::certifi=2021.10.8' \
        'conda-forge::scikit-learn=1.0.1' \
        'conda-forge::wheel=0.37.0' \
        'conda-forge::requests=2.27.1' \
        'conda-forge::urllib3=1.26.9'
      
 A few packages require pip (we won't include them in original Cocoa pip cache)
        
      $ conda activate cocoaemu
        
      $ $CONDA_PREFIX/bin/pip install --upgrade-strategy only-if-needed --no-cache-dir \
           'oauthlib==3.2.0' \
           'markdown==3.3.6' \
           'tensorflow==2.8.0' \
           'keras==2.8.0'  \
           'progressbar2==4.0.0' \
           'baccoemu==2.1.0' \
           'euclidemu2==1.2.0'
           
      $ conda deactivate  # this finish installation

Remember: With this installation method, users must activate the Conda environment whenever working with Cocoa, as shown below 
  
     $ conda activate cocoaemu
