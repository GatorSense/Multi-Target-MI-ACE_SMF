function [results, initRunTime] = miTargets(data, parameters)
%{
Input: 
*************************************************************
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
    samplePor: (0 -> 1) percentage in decimal form of positive data points used for init type 1

*************************************************************

Output:
*************************************************************
results:
    init_t: learned target signature(s)
    optObjVal: Value from objective value function of the target(s) returned
    pBagsMax: the sample from each positive bag that has the highest confidence against the target(s)
    b_mu: calculated background mean
    sig_inv_half: result from SVD calculation

%}

nBags = length(data.dataBags);
nDim = size(data.dataBags{1}, 2);
nPBags = sum(data.labels == parameters.posLabel);

%Ensure more positive bags then desired number of targets to learn
if(nPBags < parameters.numTargets)
    msg = ['You must have more positive bags than the number of targets set in the parameters' newline ...
        blanks(5) 'Number of positive bags: ' num2str(nPBags) newline ...
        blanks(5) 'Number of targets (parameters): ' num2str(parameters.numTargets)];
    error(msg);
end

%Estimate background mean and inv cov
if(parameters.globalBackgroundFlag)
    data = vertcat(data.dataBags{:});
    b_mu = mean(data);
    b_cov = cov(data)+eps*eye(size(data, 2));
else
    nData = vertcat(data.dataBags{data.labels == parameters.negLabel});
    b_mu = mean(nData);
    b_cov = cov(nData)+eps*eye(size(nData, 2));
%     b_cov = cov(nData);

end

%Whiten Data
[U, D, V] = svd(b_cov);
sig_inv_half = D^(-1/2)*U';
dataBagsWhitened = {};
for i = 1:nBags
    m_minus = data.dataBags{i} - repmat(b_mu, [size(data.dataBags{i}, 1), 1]);
    m_scale = m_minus*sig_inv_half';
    if(parameters.methodFlag)
        denom = sqrt(repmat(sum(m_scale.*m_scale, 2), [1, nDim]));
        dataBagsWhitened{i} = m_scale./denom;
    else
        dataBagsWhitened{i} = m_scale;
    end
end
pDataBags = dataBagsWhitened(data.labels ==  parameters.posLabel);
nDataBags = dataBagsWhitened(data.labels == parameters.negLabel);


if(isfield(data, 'info'))
    pBagInfo = data.info(data.labels == parameters.posLabel);
%     nBagInfo = data.info(data.labels == parameters.negLabel);
else
    pBagInfo = 'No Info';
%     nBagInfo = 'No Info';
end

%Initialize target concept
initRunTime = 0;
% status('Initializing', parameters);
if(parameters.initType == 1)
    timerVal = tic;
    [initTargets, initTargetLocation, pDataBagNumbers, initObjectiveValue] = init1(pDataBags, nDataBags, parameters);
    initRunTime = toc(timerVal);
elseif(parameters.initType == 2)
    timerVal = tic;
    [initTargets, initObjectiveValue, clustCenters] = init2(pDataBags, nDataBags, parameters);
    initRunTime = toc(timerVal);
else
    keyboard; %invalid initType
end

if(parameters.optimize)
    results = optimizeTargets(data, initTargets, parameters);
    results.methodFlag = parameters.methodFlag;  
    
else
    
    %Undo whitening
    initTargets = (initTargets*D^(1/2)*V');
    for tar = 1:parameters.numTargets
        initTargets(tar,:) = initTargets(tar,:)/norm(initTargets(tar,:));
    end
    
    results.b_mu = b_mu;
    results.b_cov = b_cov;
    results.sig_inv_half = sig_inv_half;
    results.initTargets = initTargets;
    results.methodFlag = parameters.methodFlag;
    results.numTargets = size(initTargets,1);
    results.optTargets = initTargets;
    
end

end

function [initTargets, initTargetLocation, originalPDataBagNumbers, initObjectiveValue] = init1(pDataBags, nDataBags, parameters)

    disp('Initializing Targets');

    pData = vertcat(pDataBags{:});
    initTargets = [];

    %Compute pDataConfidences
    [pDataConfidences, pDataBagNumbers] = computePDataSimilarityMatrix(pDataBags);
    originalPDataBagNumbers = pDataBagNumbers; %Keep a copy for labeling plots

    %Compute nDataConfidences
    [nDataConfidences, nDataBagNumbers] = computeNDataSimilarityMatrix(pDataBags, nDataBags);

    numPData = size(pData, 1);

    %A boolean matrix to include the sample for a target concept consideration
    includeMatrix = ones(numPData, 1);

    initTargetLocation = zeros(2, parameters.numTargets);
    initObjectiveValue = zeros(1, parameters.numTargets);

    %For all targets
    for target = 1:parameters.numTargets

        disp(['Initializing Target: ' num2str(target)]);

        objectiveValues = -100*ones(1, numPData);
        averagePBag = 100*ones(numPData, 1);
        averageNBag = 100*ones(numPData, 1);

        numTargetsLearned = target - 1;

        %After each target is calculated remove pData with that tag
        samplePBagMaxConfs = 100*ones(numPData, size(pDataBags, 2));
        %Compute objective function value for every pData sample
        for sampleNum = 1:numPData
            %Only consider the samples that don't look like targets we've already chosen
            if(includeMatrix(sampleNum))
                [objectiveValues(sampleNum), ~, samplePBagMaxConfs(sampleNum,:), averagePBag(sampleNum), averageNBag(sampleNum)] = evalObjectiveFunctionLookup(numTargetsLearned, initTargetLocation, sampleNum, pDataBags, pDataConfidences, pDataBagNumbers, nDataConfidences, nDataBagNumbers, parameters);
            end
        end

        %Take max objective value
        [initObjectiveValue(1,target), initTargetLocation(1,target)] = max(objectiveValues(:));
        %Store location and bag number for indexing in objective value function
        initTargetLocation(2,target) = originalPDataBagNumbers(initTargetLocation(1,target));

        %Extract sample at optTargetLocation
        initTarget = pData(initTargetLocation(1,target), :);
        initTarget = initTarget/norm(initTarget);

        initTargets = vertcat(initTargets, initTarget);
        numTargetsLearned = target;

        removeSimilarThresh = 1;
        %Remove similar data to target selected
        [includeMatrix, pDataBagNumbers] = removeSimilarData(pData, pDataBagNumbers, initTargetLocation, numTargetsLearned, removeSimilarThresh);

    end

end

%Initialize using K Means and picking cluster centers that maximize objective function
function [initTargets, objectiveValues, C] = init2(pDataBags, nDataBags, parameters)

    pData = vertcat(pDataBags{:});

    disp('Clustering Data');

    %Get cluster centers (C)
    [~, C] = kmeans(pData, min(size(pData, 1), parameters.numClusters), 'MaxIter', parameters.maxIter);
    
    initTargets = zeros(parameters.numTargets, size(C,2));
    numTargetsLearned = 0;
    for target = 1:parameters.numTargets
        disp(['Initializing Target: ' num2str(target)]);

        objectiveValues = zeros(1, size(C,1));
        pBagMaxConf = zeros(size(C,1), size(pDataBags, 2));
        for j = 1:size(C, 1) %if large amount of data, can make this parfor loop
            [objectiveValues(j), ~, pBagMaxConf(j,:)] = evalObjectiveFunction(pDataBags, nDataBags, C(j, :), initTargets, numTargetsLearned, parameters);
        end
        
        %Get location of max objective value
        [~, opt_loc] = max(objectiveValues);

        initTargets(target,:) = C(opt_loc, :);

        C(opt_loc,:) = 0;

        numTargetsLearned = numTargetsLearned + 1;
    end
    
    %Normalize targets
    for target = 1:parameters.numTargets
        initTargets(target,:) = initTargets(target, :) / norm(initTargets(target, :));
    end

end

%Removes potential targets that look similar to the target already chosen
function [includeMatrix, pDataBagNumbers] = removeSimilarData(pData, pDataBagNumbers, initTargetLocation, numTargetsLearned, threshold)

includeMatrix = ones(size(pData, 1), 1);

for target = 1:numTargetsLearned
    
    chosenSig = pData(initTargetLocation(target), :);

    similarity = sum(pData.*repmat(chosenSig, [size(pData, 1), 1]), 2);

    %Holds the indexes of pData that need to be excluded for learning the next target (ones should be included, zeros should be excluded)
    if(threshold == 1)
        %If threshold = 1, only remove the one datapoint in the include matrix
        includeMatrix(initTargetLocation(target), 1) = 0;
    else
        %Remove datapoints that are similar to the targets already chosen
        includeMatrix(similarity >= threshold) = 0;
    end

end

%Zero out pDataBagNumbers that correspond to the pData being removed for considerable targets (this is only computed for the most recently
%learned target signature)
if(threshold == 1)
    %If threshold = 1, only remove the one datapoint in the include matrix
    pDataBagNumbers(1,initTargetLocation(1,numTargetsLearned)) = 0;
else
    %Remove datapoints that are similar to the targets already chosen
    pDataBagNumbers(similarity >= threshold) = 0;
end

end


function [pDataConfidences, pDataBagNumbers] = computePDataSimilarityMatrix(pDataBags)

pDataBagNumbers = [];
numPBags = size(pDataBags, 2);

allPData = vertcat(pDataBags{:});
allPDataNumSamps = size(allPData,1);

%number of dimensions to the data
dataDimensions = size(pDataBags{1}, 2);

%Store the dataBag number in a vector to be able to know what sample came from what bag in the pDataConfidences, needed for calculating
%objective function value
for dataBag = 1:numPBags

    pBagNumSamps = size(pDataBags{dataBag}, 1);
    
    bagNumber = dataBag*ones(1, pBagNumSamps);
    
    pDataBagNumbers = horzcat(pDataBagNumbers, bagNumber);
        
end

%Preallocate pDataConfidences
pDataConfidences = cell(1, numPBags);

%Calculate the pDataConfidences for each dataBag
for dataBag = 1:numPBags
    
    %Individual positive bag
    pBag = pDataBags{dataBag};
    
    pBagNumSamps = size(pBag, 1);
    
    pBagReshape = reshape(pBag, [pBagNumSamps, 1, dataDimensions]);
    
    pBagCube = repmat(pBagReshape, 1, allPDataNumSamps);
    
    allPDataReshape = reshape(allPData, [1, allPDataNumSamps, dataDimensions]);
    
    allPDataCube = repmat(allPDataReshape, pBagNumSamps, 1);
    
    %Calculate confidences
    pBagConfidences = sum(pBagCube.*allPDataCube, 3);

    pDataConfidences{dataBag} = pBagConfidences;
    
end


end


function [nDataConfidences, nDataBagNumbers] = computeNDataSimilarityMatrix(pDataBags, nDataBags)

allPData = vertcat(pDataBags{:});
allNData = vertcat(nDataBags{:});
nDataBagNumbers = [];

pDataNumSamples = size(allPData,1);

%Store the dataBag number in a vector to be able to know what sample came from waht bag in the nDataConfidences, needed for calculating
%objective function value
for dataBag = 1:size(nDataBags, 2)
   
    numSamps = size(nDataBags{dataBag}, 1);
    
    %holds the bag number for all the samples in 'dataBag'
    bagNumber = dataBag*ones(1, numSamps);
    
    %Create a long vector that will be used to index nDataConfidences with the bag numbers
    nDataBagNumbers = horzcat(nDataBagNumbers, bagNumber);
    
end

%Preallocate nDataConfidences
nDataConfidences = cell(1,size(nDataBags, 2));

%Dimensionality of our data
dataDimensions = size(pDataBags{1}, 2);

%Calculate outside loop for efficiency. Does not depend on negative bag
pDataReshape = reshape(allPData, [pDataNumSamples, 1, dataDimensions]);

%Had to calculate this for each negative bag because the data was too large for matlab's array size limit
for dataBag = 1:size(nDataBags, 2)
    
    nBagNumSamples = size(nDataBags{dataBag}, 1);

    %Individual positive bag
    nBag = nDataBags{dataBag};

    %reshape to do matrix wise calculation
    nBag = reshape(nBag, [1, nBagNumSamples, dataDimensions]);

    %Set up pDataCube for matrix calculation to be of size (numSamples in allPData x numSamples in nBag)
    pDataCube = repmat(pDataReshape, 1, nBagNumSamples);

    %Set up nBagCube for matrix calculation to be of size(
    nBagCube = repmat(nBag, pDataNumSamples, 1);

    %Calculate confidences at matrix level
    nBagNDataConfidences = sum(pDataCube.*nBagCube, 3);
    
    %Store each inside a cell list
    nDataConfidences{dataBag} = nBagNDataConfidences;
    
end

end