function [results, initRunTime] = miTargets(data, parameters)
% Calls various functions to whiten data, initalize targets, and optimize 
% targets using Multiple Target Multiple Instance methodology.  
% *************************************************************
% Inputs: 
% *************************************************************
% 1) data:
%     dataBags: bagged data
%         * a positive bag should have at least one positive instance in it
%         * a negative bag should consist of all negative instances
% 
%     labels: labels for dataBags
%         * the labels should be a row vector with labels corresponding to the 
%         * parameters.posLabel and parameters.negLabel where a posLabel corresponds
%         * to a positive bag and a negLabel corresponds to a negative bag.
%         * The index of the label should match the index of the bag in dataBags
% 2) parameters:
%     numTargets: how many targets will be learned
%     initType: how the target is initialized. four possible inputs: 1, 2, 3, or 4
%     optimize: (boolean) if target signatures will be optimized or not
%     maxIter: how many possible iterations for optimizing target signature ex) 100
%     methodFlag: (boolean) Use ACE (1) or SMF (0) as similarity measure    
%     globalBackgroundFlag: (boolean) estimate the background mean and inv cov from all data or just negative bags
%     posLabel: what denotes a positive bag's label. ex) 1
%     negLabel: what denotes a negative bag's label. ex) 0
%     abs: taking absolute value of confidences
%     softmaxFlag: (boolean) Not yet implemented, keep set to 0
%     samplePor: (0 -> 1) percentage in decimal form of positive data points used for init type 1
% *************************************************************
% Output:
% *************************************************************
% 1) results:
%     optTargets: optimized learned target signatures (only returned if optimization done)
%     optObjVal: value from objective function of the target(s) returned after optimization (only returned if optimization done)
%     b_mu: calculated background mean
%     b_cov: calculated background covariance from SVD
%     sig_inv_half: result from SVD calculation
%     initTargets: initialized target signatures (will always be the number set in setParameters.m)
%     methodFlag: method used. ACE (1) vs SMF (0)
%     numTargets: number of learned targets.
%     initTargetsLocation: ONLY if init 1 is specified. [num_initializedtargets, 2],
%           first column is index within bag, second column index of bag
%     optTargetsLocation: ONLY if init 1 is specified. [num_optimizedtargets, 2],
%           first column is index within bag, second column index of bag
% ------------------------------------------------------------------------

%Ensure more positive bags then desired number of targets to learn
nPBags = sum(data.labels == parameters.posLabel);
if(nPBags < parameters.numTargets)
    msg = ['You must have more positive bags than the number of targets set in the parameters' newline ...
        blanks(5) 'Number of positive bags: ' num2str(nPBags) newline ...
        blanks(5) 'Number of targets (parameters): ' num2str(parameters.numTargets)];
    error(msg);
end

% 1) Whiten Data
[dataBagsWhitened, dataInfo] = whitenData(data, parameters);
pDataBags = dataBagsWhitened.dataBags(data.labels ==  parameters.posLabel);
nDataBags = dataBagsWhitened.dataBags(data.labels == parameters.negLabel);

% 2) Initialize target signatures and maximize objective function
initRunTime = 0;
if(parameters.initType == 1)
    % Initialize by searching all positive instances and greedily selects
    % instances that maximizes objective function. 
    timerVal = tic;
    [initTargets, initTargetLocation, pDataBagNumbers, initObjectiveValue] = init.init1(pDataBags, nDataBags, parameters);
    parameters.initTargetsLocation = transpose(initTargetLocation); % adds to parameter list for optimization
    initRunTime = toc(timerVal);
elseif(parameters.initType == 2)
    % Initialize by K-means cluster centers and greedily selecting cluster
    % center that maximizes objective function. 
    timerVal = tic;
    [initTargets, initObjectiveValue, clustCenters] = init.init2(pDataBags, nDataBags, parameters);
    initRunTime = toc(timerVal);
else
    disp('Invalid initalization parameter. Options are 0, 1, or 2.')
    return
end

% 3) Optimize target concepts 
if parameters.optimize == 0
    % Do not optimize targets - will return the initialized targets
    results = opt.nonOptTargets(initTargets, parameters, dataInfo);
elseif parameters.optimize == 1
    % optmize targets using MT MI methodology
    results = opt.optimizeTargets(initTargets, pDataBags, nDataBags, parameters, dataInfo);
else
    disp('Invalid optimize parameter. Options are 0 or 1.')
    return
end

end