# Multi-Target MI-ACE and MI-SMF:
**Multiple Target Multiple Instance Adaptive Cosine Estimator and Spectral Match Filter for Target Detection Using Uncertainly Labeled Data**

_James Bocinsky, Alina Zare, and Susan Meerdink_

If you use this code, cite it: To be completed...

[[`BibTeX`](#Citing)]

In this repository, we provide the papers and code for the Multi-Target MI-ACE and MI-SMF Algorithm.

## Installation Prerequisites

This code uses MATLAB Statistics and Machine Learning Toolbox,
MATLAB Optimization Toolbox and MATLAB Parallel Computing Toolbox.

## Cloning

To recursively clone this repository using Git to include the hyperspectral demo submodule use the following command:

     git clone --recursive https://github.com/GatorSense/Multi-Target-MI-ACE_SMF.git

## Demo and Example

For Demo, run `simulateHyperspectral.m` in MATLAB.
For Example, check out code in exampleMain folder. 

## Main Functions

Run the algorithm using the following function:

```results = miTargets(data, parameters);```


## Inputs

    data.dataBags: cell list with positive and negative bags (1xNumBags). Each cell contains a single bag in the form a (numInstances x instanceDimensionality) matrix
        * a positive bag should have at least one positive instance in it
        * a negative bag should consist of all negative instances

    data.labels: labels for dataBags (1xnumBags)
        * the labels should be a row vector with labels corresponding to the 
        * parameters.posLabel and parameters.negLabel where a posLabel corresponds
        * to a positive bag and a negLabel corresponds to a negative bag.
        * The index of the label should match the index of the bag in dataBags

    parameters: call setParameters.m function to set parameters
    

## Parameters
The parameters can be set in the following function:

```setParameters.m```

The parameters is a MATLAB structure with the following fields:
1. methodFlag: Determine similarity method 
     * 0: Spectral Match Filter (SMF)
     * 1: Adaptive Cosine Estimator (ACE)
2. numTargets: Number of initialized targets
3. initType:  Set Initialization method used to obtain initalized targets
     * 1: Searches all positive instances and greedily selects instances that maximize objective.  
     * 2: K-Means clusters positive instances and greedily selects cluster centers that maximize objective.
4. optimize: Determine whether to optimize target signatures
     * 0: Do not optimize target signatures, only returns initialized targets
     * 1: Optimize target signatures using MT MI
5. maxIter: Max number of optimization iterations
6. globalBackgroundFlag: Determines how the background mean and inverse covariance are calcualted
     * 0: mean and cov calculated from all data (positive and negative data)
     * 1: mean and cov calculated from only negative bags
7. posLabel: Label used for positive bags
8. negLabel: Label used for negative bags
9. numClusters: Number of clusters used for K-Means (only affects if initType 2 used)
10. alpha: Uniqueness term weight in objective function
     * set to 0 if you do not want to use the term

## Inventory

* Note: some of the functions from the Hyperspectral_Data_Simulation are used to run the demo. Check out the repo here: [[`Hyperspectral_Data_Simulation Repository`](https://github.com/GatorSense/Hyperspectral_Data_Simulation)]

```
https://github.com/GatorSense/Multi-Target-MI-ACE_SMF

└── root dir
    ├── LICENSE  //MIT license
    ├── README.md  //this file
    └── algorithm  //algorithm functions
        ├── evalObjectiveFunction.m //evaluates the objective function given a candidate target signature
        ├── evalObjectiveFunctionLookup.m //evaluates the objective function for initialization using a precomputed similarity matrix of data
        ├── miTargets.m  //runs algorithm
        ├── optimizeTargets.m  //optimizes target signatures
        └── setParameters.m  //sets parameters for algorithm
        └── detectors  //algorithm functions
            ├── ace_det.m  //compute ACE
            └── smf_det.m  //compute SMF
    └── exampleMain  //template for your own development
        ├── bagData.m  //template to bag data
        ├── bagHyperspectral.m //code that bags example data
        ├── example_data.csv // csv containing example hyperspectral data
        ├── splitTrainTest.m // code that split example data into training and validation using KFold cross validation
        ├── roc_example_results.fig //matlab figure displaying roc results from example data
        └── exampleMain.m  //template to set up data and run algorithm
    └── hyperspectralDataSimulationCode  //simulation demo
        ├── parameters.m  //set parameters for simulation demo
        └── simulateHyperspectral.m  //run simulation demo
        
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2019 J. Bocinsky and A. Zare. All rights reserved.

## <a name="Citing"></a>Citing Multi-target MI-ACE and MI-SMF

If you use the Multi-target MI-ACE or Multi-target MI-SMF algorithm, please cite the following reference using the following BibTeX entries.
```
@MastersThesis{Bocinsky2019Thesis,
author = {James Bocinsky},
title = {Learning Multiple Target Concepts from Uncertain, Ambiguous Data Using the Adaptive Cosine Estimator and Spectral Match Filter},
school = {Univ. of Florida},
year = {2019},
address = {Gainesville, FL},
month = {May},
}
```       
```
@Article{Meerdink2019DevelopingSpectralLibrariesMTMIACE,  
Title = {Developing Spectral Libraries Using Multiple Target Multiple Instance Adaptive Cosine/Coherence Estimator}, 
Author = {S. Meerdink and J. Bocinsky and E. Wetherley and A. Zare and C. McCurley and P. Gader},  
Booktitle={10th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS)}
Year = {2019},  
month={Sep.},  
pages={1-5},
}
```
