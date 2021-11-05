# MGCosmoPop
This package implements a hierarchical bayesian framework for constraining the standard cosmological parameters (Hubble constant and Dark Matter density) and modified Gravitational Wave propagation parameters together with the Binary Black Hole (BBH) population parameters (mass function, merger rate density, spin distribution)


Developed by [Michele Mancarella](<https://github.com/Mik3M4n>).


Here we show the analysis of the GWTC-2 catalog  with a BBH population model given by a broken power law mass distribution using MGCosmoPop, and compare to the result of the LVC. See [the LVC paper](https://arxiv.org/abs/2010.14533) for details. We compare the LVC result (blue) to the result obtained with this code using the LVC injections for computing selection effects (green) and using this code and our own injections (red).

![alt text](https://github.com/CosmoStatGW/MGCosmoPop/blob/master/R0_lambda_alpha1_alpha2_beta_deltam_ml_mh_b_corner_local_LVC.png?raw=true)


## Summary


* [Overview and code organisation](https://github.com/CosmoStatGW/MGCosmoPop#Overview-and-code-organisation)
* [Data](https://github.com/CosmoStatGW/MGCosmoPop#Data)
* [Usage](https://github.com/CosmoStatGW/MGCosmoPop#Usage)




## Overview and code organisation

### General structure
The code implements the hierarchical framework in an object-oriented fashion. 
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

#### Mass functions
* Truncated power law
* Truncated power law with smoothed edges
* Broken power law

#### Rate evolution
* Power law : R(z) = R_0 * (1+z)^\lambda
* Madau-Dickinson rate

#### Spin distributions
* (Uncorrelated) Gaussian model for chi\_1, chi\_2

### Extensions

To add models, one should inherit from the corresponding ABC and implement the required functions. 
To add populations, one should implement a new module (e.g. : PBHs )


## Data
### GWTC-2

O1-O2 and O3a are supported. The corresponding posterior samples should be placed under data/O1O2 and data/O3a respectively.

### Mock datasets
Coming soon...

## Usage

An introductory notebook is available in notebooks/


