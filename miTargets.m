function results = miTargets(data, parameters)
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

%Estimate background mean and inv cov
if(parameters.globalBackgroundFlag)
    data = vertcat(data.dataBags{:});
    b_mu = mean(data);
    b_cov = cov(data)+eps*eye(size(data, 2));
else
    nData = vertcat(data.dataBags{data.labels == parameters.negLabel});
    b_mu = mean(nData);
    b_cov = cov(nData)+eps*eye(size(nData, 2));
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

%Initialize target concept
status('Initializing', parameters);
if(parameters.initType == 1)
    [init_t, optObjVal, pBagsMax] = init1(pDataBags, nDataBags, parameters);
elseif(parameters.initType == 2)
    [init_t, optObjVal, pBagsMax] = init2(pDataBags, nDataBags, parameters);
elseif(parameters.initType == 3)
    [init_t, optObjVal, pBagsMax] = init3(pDataBags, nDataBags, parameters);
elseif(parameters.initType == 4)
    [init_t, optObjVal, pBagsMax] = init4(pDataBags, nDataBags, parameters);
else
    keyboard; %invalid initType
end

if parameters.optimize
    results = optimizeDict(data, init_t, parameters);
else
    results.optObjVal = optObjVal;
    results.pBagsMax = pBagsMax;
    results.b_mu = b_mu;
    results.sig_inv_half = sig_inv_half;
    results.init_t = init_t;
end

end

function [init_t, optObjVal, pBagsMax] = init1(pDataBags, nDataBags, parameters)

pData = vertcat(pDataBags{:});
temp = randperm(size(pData, 1));
samplepts = round(size(pData, 1)*parameters.samplePor);
pData_reduced = pData(temp(1:samplepts), :);
tempObjVal = zeros(1, size(pData_reduced, 1));
for j = 1:size(pData_reduced, 1) %if large amount of data, can make this parfor loop
    optTarget = pData_reduced(j, :);
    tempObjVal(j) = evalObjectiveWhitened(pDataBags, nDataBags, optTarget, parameters.softmaxFlag);
end
[~, opt_loc] = max(tempObjVal);
optTarget = pData_reduced(opt_loc, :);
optTarget = optTarget/norm(optTarget);
init_t = optTarget;
[optObjVal, pBagsMax] = evalObjectiveWhitened(pDataBags, nDataBags, optTarget, parameters.softmaxFlag);

end

function [init_t, optObjVal, pBagsMax] = init2(pDataBags, nDataBags, parameters)

%Select data point with smallest cosine of the vector angle to background data
pData = vertcat(pDataBags{:});
nData = vertcat(nDataBags{:});
nDim = size(pData, 2);
pdenom = sqrt(repmat(sum(pData.*pData, 2), [1, nDim]));
ndenom = sqrt(repmat(sum(nData.*nData, 2), [1, nDim]));
pData = pData./pdenom;
nData = nData./ndenom;

nDataMean = mean(nData);

tempObjVal = sum(pData.*repmat(nDataMean, [size(pData, 1), 1]), 2);
[~, opt_loc] = min(tempObjVal);
optTarget = pData(opt_loc, :);
optTarget = optTarget/norm(optTarget);
init_t = optTarget;
[optObjVal, pBagsMax] = evalObjectiveWhitened(pDataBags, nDataBags, optTarget, parameters.softmaxFlag);

end

function [init_t, optObjVal, pBagsMax] = init3(pDataBags, nDataBags, parameters)

%Run K-means and initialize with the best of the cluster centers
pData = vertcat(pDataBags{:});

[idx, C] = kmeans(pData, min(size(pData, 1), parameters.initK), 'MaxIter', parameters.maxIter);

tempObjVal = zeros(1, length(unique(idx)));
parfor j = 1:size(C, 1) %if large amount of data, can make this parfor loop
    optTarget = C(j, :)/norm(C(j, :));
    tempObjVal(j) = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, optTarget, parameters);
end

init_t = zeros(parameters.numTargets,size(C,2));

for target = 1:parameters.numTargets
    %%%%%update eval function
    [~, opt_loc] = max(tempObjVal);
    [~, min_loc] = min(tempObjVal);
    init_t(target,:) = C(opt_loc, :) / norm(C(opt_loc, :));
    tempObjVal(opt_loc) = tempObjVal(min_loc);
end

[optObjVal, pBagsMax] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, init_t, parameters);

end

function [init_t, optObjVal, pBagsMax, tags] = init4(pDataBags, nDataBags, parameters)

pData = vertcat(pDataBags{:});
optObjVal = zeros(parameters.numTargets,1);
pBagsMax = cell(parameters.numTargets,1);
targets = [];

for numTar = 1:parameters.numTargets
    tempObjVal = [];
    for point = 1:size(pData, 1)
        testTarget = pData(point, :);
        testGroup = vertcat(targets,testTarget);
        tempObjVal(point) = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, testGroup, parameters);
    end
    
    [optObjVal(numTar), opt_loc] = max(tempObjVal);
    optTarget = pData(opt_loc, :);
    optTarget = optTarget/norm(optTarget);
    targets = vertcat(targets, optTarget);
    
    pData = pData([1:opt_loc-1 opt_loc+1:end], :);
end

for target = 1:parameters.numTargets
    [~, pConfMax] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, targets(target, :), parameters);
    pBagsMax(target,1) = pConfMax;
end

init_t = targets;

end

