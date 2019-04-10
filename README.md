# Multi-Target MI-ACE and MI-SMF

Multi Target - Multiple Instance - Adaptive Cosine Estimator and Multiple Instance - Spectral Match Filter

Matlab Implementation

****************************************************************

NOTE: If the Multi-target MI-ACE and MI-SMF Algorithm is used in any publication or presentation, the following reference must be cited: 
J. Bocinsky, A. Zare

NOTE: If this code is used in any publication or presentation, the following reference must be cited:
J. Bocinsky, A. Zare. (2019, April 09). GatorSense/MTMIACE: Version 1 (Version v1.0).

****************************************************************

A hyperspectral simulation demo is provided in hyperspectralDataSimulation. To run the demo run:
    
    simulateHyperspectral.m
    
If you are using this algorithm on your own data, please see the template code provided in exampleMain.

The command to run algorithm:

    results = miTargets(data, parameters);

Input: 

    data.dataBags: bagged data
        * a positive bag should have at least one positive instance in it
        * a negative bag should consist of all negative instances

    data.labels: labels for dataBags
        * the labels should be a row vector with labels corresponding to the 
        * parameters.posLabel and parameters.negLabel where a posLabel corresponds
        * to a positive bag and a negLabel corresponds to a negative bag.
        * The index of the label should match the index of the bag in dataBags

    parameters: see algorithm/setParameters.m for description

Output:

    results.init_t: learned target signature(s)
    results.optObjVal: Value from objective value function of the target(s) returned
    results.pBagsMax: the sample from each positive bag that has the highest confidence against the target(s)
    results.b_mu: calculated background mean
    results.sig_inv_half: result from SVD calculation           
