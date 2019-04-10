# Multi-Target MI-ACE and MI-SMF

Multi Target - Multiple Instance - Adaptive Cosine Estimator and Multiple Instance - Spectral Match Filter

Matlab Implementation

****************************************************************

NOTE: If the Multi-target MI-ACE and MI-SMF Algorithm is used in any publication or presentation, the following reference must be cited: 
J. Bocinsky, A. Zare

NOTE: If this code is used in any publication or presentation, the following reference must be cited:
J. Bocinsky, A. Zare. (2019, April 09). GatorSense/MTMIACE: Version 1 (Version v1.0).

****************************************************************


The command to run:

results{iter} = miTargets(data, parameters);

Input: 

data:

    dataBags: bagged data
        * a positive bag should have at least one positive instance in it
        * a negative bag should consist of all negative instances

    labels: labels for dataBags
        * the labels should be a row vector with labels corresponding to the 
        * parameters.posLabel and parameters.negLabel where a posLabel corresponds
        * to a positive bag and a negLabel corresponds to a negative bag.
        * The index of the label should match the index of the bag in dataBags

parameters:
    
    numTargets: how many targets will be learned
    initType: how the target is initialized. four possible inputs: 1, 2, 3, or 4
    optimize: (boolean) if target signatures will be optimized or not
    maxIter: how many possible iterations for optimizing target signature ex) 100
    methodFlag: (boolean) Use ACE (1) or SMF (0) as similarity measure    
    globalBackgroundFlag: (boolean) estimate the background mean and inv cov from all data or just negative bags
    posLabel: what denotes a positive bag's label. ex) 1
    negLabel: what denotes a negative bag's label. ex) 0
    abs: taking absolute value of confidences
    softmaxFlag: (boolean) Not yet implemented, keep set to 0


Output:

results:
    
    init_t: learned target signature(s)
    optObjVal: Value from objective value function of the target(s) returned
    pBagsMax: the sample from each positive bag that has the highest confidence against the target(s)
    b_mu: calculated background mean
    sig_inv_half: result from SVD calculation

      You will need to bag your data as follows:

          data.dataBags: bagged data
              * a positive bag should have at least one positive instance in it
              * a negative bag should consist of all negative instances

          data.labels: labels for dataBags
              * the labels should be a row vector with labels corresponding to the
              * parameters.posLabel and parameters.negLabel where a posLabel corresponds
              * to a positive bag and a negLabel corresponds to a negative bag.
              * The index of the label should match the index of the bag in dataBags
            
