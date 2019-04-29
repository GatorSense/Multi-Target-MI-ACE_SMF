# Multi-Target MI-ACE and MI-SMF:
**Multiple Target Multiple Instance Adaptive Cosine Estimator and Spectral Match Filter for Target Detection Using Uncertainly Labeled Data**

_James Bocinsky and Alina Zare_

If you use this code, cite it: To be completed...

[[`BibTeX`](#Citing)]

In this repository, we provide the papers and code for the Multi-Target MI-ACE and MI-SMF Algorithm.

## Installation Prerequisites

This code uses MATLAB Statistics and Machine Learning Toolbox,
MATLAB Optimization Toolbox and MATLAB Parallel Computing Toolbox.

## Demo

Run `simulateHyperspectral.m` in MATLAB.

## Main Functions

The algorithm runs using the following function:

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

    parameters: call setParameters.m function and see file to set parameters
    

## Parameters
The parameters can be set in the following function:

```setParameters.m```

The parameters is a MATLAB structure with the following fields:
1. methodFlag: ACE = 1, SMF = 0
2. numTargets: Number of initialized targets
3. initType: (1) Search through all positive instances or (2) Search through K-Means cluster centers of positive instances
4. optimize: Optimize targets or not
5. maxIter: Max number of optimization iterations
6. globalBackgroundFlag: (1) Use all the data (including positive bags) or (0) just the negative instances to compute background statistics
7. posLabel: Label used for positive bags
8. negLabel: Label used for negative bags
9. numClusters: Number of clusters if using initType (2)
10. uniqueTargets: (1) Use uniqueness term or (0) not
11. alpha: Uniqueness term weight in objective function

*Parameters can be modified by users in [parameters] = setParameters() function.*

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
        └── exampleMain.m  //template to set up data and run algorithm
    └── hyperspectralDataSimulationCode  //simulation demo
        ├── parameters.m  //set parameters for simulation demo
        └── simulateHyperspectral.m  //run simulation demo
        
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2019 J. Bocinsky and A. Zare. All rights reserved.

## <a name="Citing"></a>Citing Multi-target MI-ACE and MI-SMF

If you use the Multi-target MI-ACE or Multi-target MI-SMF algorithm, please cite the following reference using the following BibTeX entries. To be completed...
```
@article{Bo2019multi,
  title={Enter Here},
  author={Bocinsky, James},
  journal={arXiv preprint arXiv:Enter Here},
  year={2019}
}
```       
