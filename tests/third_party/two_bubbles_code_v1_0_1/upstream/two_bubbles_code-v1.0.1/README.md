# Running two bubble simulation code and gw spec code

## Simulation:

The simulation code hyper_bubbles.py calculates the field evolution and energy density for a given gamma and lambdabar. This version of the code takes into account the evolution of the whole field of the two bubbles.

The code is written in Python and it requires version 3 of Python and the installation of [CosmoTransitions](https://github.com/clwainwright/CosmoTransitions?tab=readme-ov-file).

The code can be run with  
```
python hyper_bubbles.py ${lambda_bar} ${gamma} ${path_to_save}
```
where lambda_bar lies in the range (0,1) and gamma>1. Gamma is the Lorentz contraction of the bubble wall at collision and lambda_bar describes the thickness of the bubble. The parameter path_to_save specifies the location where data is to be saved. 

The code produces a plot of the field evolution and a pickle file containing the data values_lambda={lambdabar}_gamma={gamma}_half.pickle. Generally, do not rename pickle files as the gravitational wave spectrum finds the file based on the name given by hyper_bubbles.py. 

The pickle files contain:
- tuple of (info, phi_to_save, z, phi_mid, energy_density)
  - info is another tuple containing constants used in simulation: (lambda_bar, gamma, d, ds, n_z, how_often_ds, r_info)
    - r_info is another tuple that contains (initial radius of bubble, inner radius, outer radius)


The distance d between the bubbles is calculated from gamma. The code calculates the spacings ds and dz automatically, but they can be changed manually as well
```
python hyper_bubbles.py ${lambda_bar} ${gamma} ${path_to_save} half ${dztimes} ${ds}
```
Here ds is the spacing in direction s, dztimes redefines the automatically calculated value of dz to be dztimes*dz.

The saved data is downsampled in coordinate s by how_often_ds (=5) defined in the code.

The simulation returns:
- field evolution plot: gamma={gamma}_lambda_bar={lambda_bar}.png
- pickle file: values_lambda={lambda_bar}_gamma={gamma}_half.pickle
  - contents of pickle file: tuple of (info, phi_to_save, z, phi_mid, energy_density).  
  - info is another tuple containing constants used in simulation: (lambda_bar, gamma, d, ds, n_z, how_often_ds, r_info)
    - r_info is another tuple that contains (initial radius of bubble, inner radius, outer radius)

The code is also able to run the evolution of one bubble, with
```
python hyper_bubbles.py ${lambda_bar} ${gamma} ${path_to_save} one
```

Other parameters can be added at the end of the command when testing. Tests so far 'energy_density_comparison', 'energy_density_convergence' and 'gamma_test'

## Fourier

The Fourier mode expansion code is contained in the fourier directory.

The cython file, fou, is built with
```
python setup.py build_ext --inplace
```

Once this is built, the code can be run on the output pickle file of the simulation code, using
```
python fourier.py ${lambdabar} ${gamma} ${path_to_data}
```
The result is a pickle file containing the mode expansion coefficients of the field. The degree of downsampling in the s and z directions is determined by every_s and every_z within the code.

## Gw spec

The code calculating the gravitational wave spectrum of a field profile is in the gw_spec directory. It calculates the gravitational wave spectrum for a given lambda_bar and gamma, using the results from hyper_bubbles.py at the same parameter values.

The calculation of the integrals uses a cython file, u_integrand, which must be built with
```
python setup.py build_ext --inplace
```

The gravitational wave integral code, gw_main_run.py that calls the file gw_integrator.py, can be run with following command (using Xfaulthandler is optional)
```
python -Xfaulthandler gw_main_run.py ${lambdabar} ${gamma} ${ID} ${NCORES} ${path_to_data}
```
Here NCORES corresponds to the amount of points $\omega$ and ID to the index of the current $\omega$ in calculation. ID should be in the range 0 to NCORES-1. NCORES=60 is a well functining value. Every index ID needs to be run separately. This will be very slow if run in serial but can be parallelised, for example on a cluster each ID can be queued up as a separate job.

As before with hyper_bubbles.py, if a different spacing from default is desired, the code can be run with
```
python -Xfaulthandler gw_main_run.py ${lambdabar} ${gamma} ${ID} ${NCORES} ${path_to_data} half ${dztimes} ${ds}
```
This presumes that a simulation pickle file with these specifications already exists.

It is possible to add these optional parameters at the end of the command:
- 'print_sub': prints subintegral results to file and to stdout
- 'plot_k': plots the k integrand

The range of $\omega$ is calculated automatically by the code. The smallest value is set by $\pi/2/$[size of the simulation box in z direction], and the largest value by min(10*[mass at true vacuum], $\pi/dz$). 

After running the NCORES amount of calculations, the code has returned some important simulation parameters in file (generated by run ID=0), and a number of txt files equal to NCORES. The files are named according to the convention:
- constants_lambda={lambda_bar}_gamma={gamma}_half.txt
- gw_results_whole_lambda={lambda_bar}_gamma={gamma}_w{ID}.txt, containing tab separated values $\omega$ and $dE/d\log(\omega)$

It is suggested to cat these different ID results to a single file by hand, e.g.
```
cat gw_results_whole_lambda={lambda_bar}_gamma={gamma}_w* > gw_results_whole_lambda={lambda_bar}_gamma={gamma}.txt
```

### Fit to the gravitational wave spectrum

The high-frequency power law of a gravitational wave spectrum can be found and the spectrum including the fit can be plotted with gw_fit.py, found in the gw_spec directory.

The fitting code takes as input the gravitational wave spectrum (combined into one file) and the parameters file output by gw_main_run.py. Run with
```
python gw_fit.py data-file-1 params-file-1 ... data-file-n params-file-n
```

# Testing files

## Simulation test

In testing folder is dynamics_tests.sh which runs interactively all the most important tests made so far.

Run with
```
sh dynamics_tests.sh
```

# Example

This section gives a real example of how to regenerate the results.
For e.g. lambda=0.01, gamma=4, the first step is to run hyper_bubbles.py.
```
python hyper_bubbles.py 0.01 4 ./results/
```
In this example, the files are saved in folder ./results/.

Presuming that the cython file has been set up, the gravitational wave spectrum calculation can be performed with gw_main_run.py.
```
for id in {0..59}
do
  python -Xfaulthandler gw_spec/gw_main_run.py 0.01 4 $id 60 ./results/
done
```
Depending on where this code is run, some parallelisation is suggested.

Combine files with 
```
cat ./results/gw_results_whole_lambda=0.01_gamma=4_w* > ./results/gw_results_whole_lambda=0.01_gamma=4.txt
```
Plot results and calculate fit parameters with
```
python gw_spec/gw_fit.py ./results/gw_results_whole_lambda=0.01_gamma=4.txt ./results/constants_lambda=0.01_gamma=4_half.txt
```
