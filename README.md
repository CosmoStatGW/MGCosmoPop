# MGCosmoPop
This package implements a hierarchical bayesian framework for constraining the standard cosmological parameters (Hubble constant and Dark Matter density) and modified Gravitational Wave propagation parameters together with the Binary Black Hole (BBH) population parameters (mass function, merger rate density, spin distribution)


Developed by [Michele Mancarella](<https://github.com/Mik3M4n>).

If using this code, please cite this repository: [![DOI](https://zenodo.org/badge/425232654.svg)](https://zenodo.org/badge/latestdoi/425232654) , 
and the paper [Cosmology and modified gravitational wave propagation from binary black hole population models](<https://arxiv.org/abs/2112.05728>). Bibtex:

```
@article{Mancarella:2021ecn,
    author = "Mancarella, Michele and Genoud-Prachex, Edwin and Maggiore, Michele",
    title = "{Cosmology and modified gravitational wave propagation from binary black hole population models}",
    eprint = "2112.05728",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "12",
    year = "2021"
}
```

The data products associated to the paper (injections used for computation of the selection effects in GWTC-3, mock datasets and injections) are avaliable at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6461447.svg)](https://doi.org/10.5281/zenodo.6461447)



## Summary


* [Overview and code organisation](https://github.com/CosmoStatGW/MGCosmoPop#Overview-and-code-organisation)
* [Installation](https://github.com/CosmoStatGW/MGCosmoPop#Installation)
* [Data](https://github.com/CosmoStatGW/MGCosmoPop#Data)
* [Usage](https://github.com/CosmoStatGW/MGCosmoPop#Usage)
* [Example](https://github.com/CosmoStatGW/MGCosmoPop#Example)

## Installation

 
### Using conda
 
 Create the dedicated environment:
 ```
 conda create -y --name gwstat python=3.7
 conda activate gwstat
 ```
 
 Clone the repo:
 
 ```
 git clone https://github.com/CosmoStatGW/MGCosmoPop.git
 cd MGCosmoPop
 ```
 
 Install dependencies and package
 
```
conda install -y -c conda-forge --file requirements_conda.txt  
pip install .
```


### Manually, without cloning
Install manually the dependencies in requirements.txt

Then

 ```
pip install git+https://github.com/CosmoStatGW/MGCosmoPop
 ```

## Overview and code organisation

### General structure
The code implements the hierarchical framework in an object-oriented way. 
For the moment, a single population of astrophysical BBHs is present. The code is however ready to support multiple populations, which should be implemented inheriting from the Abstract Base Class ```ABSpopulation```


The organisation of the code is the following:

```bash
MGCosmoPop/MGCosmoPop/
					├── cosmology/ 
							 a class Cosmo implementing cosmology-related functions
					├── dataStructures/
							 One abstract base class for data and classes for reading and using mock data and data from the O1-O2 and O3a observing runs. Classes for reading and using injections to compute selection effects are also there
					├── mock/
							 Tools to generate mock datasets and injections
					├── population/
							 Classes for implementing the population function. Described below separately
					├── posteriors/
							 Classes implementing likelihood, posterior and selection effects
					├── sample/
							 MCMC tools
						
```		

### The population model

The key module for the population function(s) is ```population```, organised as follows:

```bash
population/
		├── ABSpopulation.py
				 Abstract Base Classes for describing a population. Contains three ABCs: 
					i) Population, that requires to implement a (log) differential rate log(dR/dm1 dm2 )  
					ii) RateEvolution requiring to implement the (log) differential log ( dN/dV dt )  = log ( R(z) ) 
					iii) BBHdistfunction, used to implement the mass and spin distribution. 
						This requires to implement a (log) probability distribution log p(m1, m2) or log p(chi1, chi2)
		├── allPopulations.py
				collects the differential rates from all populations, adds the volume element jacobian, 
				the jacobian between source and detector frame variables, observation time, 
				and yields the full population function dN/dtheta (theta = {m1_det, m2_det , dL, chi_1, chi_2...})
		├── astro/ 
			 population of astrophysical black holes. 
			 	├── astroPopulation.py
			 			population function dR/dm1dm2
			 	├── astroMassDistribution.py 
			 			mass function p(m1, m2)
				├── astroSpinDistribution.py 
			 			spin distribution
			 	├── rateEvolution.py
			 			merger rate density 
					
```		

An explanatory notebook on how to set up a model is provided in the notebook folder.

## Supported models

The following models are implemented: 

### Astrophysical black holes

The astrophysical BH population is defined by three base ingredients: mass distribution, spin distribution, and merger rate evolution. Each one is implemented in a specific object.

#### Mass functions (primary mass)
* Truncated power law
* Truncated power law with smoothed edges
* Broken power law
* (From v1.1) Power law + peak model 
* (From v1.1) Neutron stars: uncorrelated gaussian
* (From v1.1) Neutron stars: uncorrelated flat

#### Rate evolution
* Power law : R(z) = R_0 * (1+z)^\lambda
* Madau-Dickinson rate

#### Spin distributions
* (Uncorrelated) Gaussian model for chi\_1, chi\_2
* (From v1.1) "Default" spin model


### Extensions

To add models, one should inherit from the corresponding ABC and implement the required functions. 
To add populations, one should implement a new module (e.g. : PBHs )


## Data

Data for O1-O2, O3a, and O3b can be downloaded from the LVK data releases. The data products associated to the paper (injections used for computation of the selection effects in GWTC-3, mock adasets and injections) are avaliable at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6461447.svg)](https://doi.org/10.5281/zenodo.6461447).

All data can be downloaded running the script ```getData.sh```:

```
chmod +x getData
./getData.sh
```



### GWTC-3

O1-O2 O3a, and O3b are supported. The corresponding posterior samples should be placed under data/O1O2, data/O3a, data/O3b respectively.

### Injections
Tools for generating injections are in mock/ . Examples of configuration files to generate injections for O3a/O3b are provided. After editing the config file, run:

```
python generateInjections.py --config=configInjections_O3b.py --fout=<name_of_output_folder> 
```

The corresponding file is saved in the output folder as ```selected.h5 ```. It can be loaded using the object ```GWMockInjectionsData ``` in ```dataStructures ```:

```
file_name  = os.path.join(<name_of_output_folder> ,'selected.h5' )
myInjections = GWMockInjectionsData(file_name ,  Tobs=Tobs )
```

where ```Tobs ``` is the duration in years of the observing run.

## Usage

An introductory notebook is available in notebooks/

A module to run a MCMC analysis is available in ```sample/``` . To run on a cluster, edit the configuration file as in the template ```config_template.py``` (explanations in the file) with the option ```parallelization='mpi'```. Then:

```
fout = myMCMC
mkdir $fout

srun  python runMCMC.py --config=config_template --fout=$fout
```

## Example

Here we show the analysis of the GWTC-2 catalog  with a BBH population model given by a broken power law mass distribution using MGCosmoPop, and compare to the result of the LVC. See [the LVC paper](https://arxiv.org/abs/2010.14533) for details. We compare the LVC result (blue) to the result obtained with this code using the LVC injections for computing selection effects (green) and using this code and our own injections (red).

![alt text](https://github.com/CosmoStatGW/MGCosmoPop/blob/master/R0_lambda_alpha1_alpha2_beta_deltam_ml_mh_b_corner_local_LVC.png?raw=true)

