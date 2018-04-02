function results = optimizeDict(data, init_t, parameters)

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

for target = 1:parameters.numTargets
    [optObjVal(target), ~] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, init_t(target, :), parameters);
end
[~, pBagsMax] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, init_t, parameters);
while(size(pBagsMax,2) < parameters.numTargets)
    pBagsMax{end + 1} = [];
end
    
optTarget = init_t;

status('Optimizing', parameters);

%Precompute term 2 in update equation
nMean = zeros(length(nDataBags), nDim);
for j = 1:length(nDataBags)
    nData = nDataBags{j};
    nMean(j, :) = mean(nData)';
end
nMean = mean(nMean);

%Train target signature
iter = 1;
continueFlag = 1;
objTracker(1).val = optObjVal;
objTracker(1).target = optTarget;

while(continueFlag && iter < parameters.maxIter)
    
    for target = 1:parameters.numTargets
        if(size(pBagsMax{target},1) > 1)
            pMean = mean(pBagsMax{target});
        else
            pMean = pBagsMax{target};
        end
        t = pMean - nMean;
        if(t ~= 0)
            optTarget(target, :) = t/norm(t);
        end
    end

    %Update Objective and Determine the max points in each bag
    for target = 1:parameters.numTargets
        [optObjVal(target), ~] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, optTarget(target, :), parameters);
    end
    [~, pBagsMax] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, optTarget, parameters);
    while(size(pBagsMax,2) < parameters.numTargets)
        pBagsMax{end + 1} = [];
    end

    iter = iter + 1;

    if(objTracker(iter-1).val == optObjVal)
        if(~sum(abs(objTracker(iter-1).target - optTarget)))
            continueFlag = 0;
            disp(['Stopping at iter: ', num2str(iter-1)]);
        end;
    end

    if(~isnan(any(optObjVal)) && ~any(isnan(any(optTarget))))
        objTracker(iter).val = optObjVal;
        objTracker(iter).target = optTarget;
    else
        objTracker(iter).val = objTracker(iter-1).val;
        objTracker(iter).target = objTracker(iter-1).target;
    end

end

%Undo whitening
optTarget = (optTarget*D^(1/2)*V');
optTarget = optTarget/norm(optTarget);
init_t = (init_t*D^(1/2)*V');
init_t = init_t/norm(init_t);

results.optDict = optTarget;
results.optObjVal = optObjVal;
results.b_mu = b_mu;
results.sig_inv_half = sig_inv_half;
results.init_t = init_t;
results.methodFlag = parameters.methodFlag;
results.numTargets = parameters.numTargets;
results.abs = parameters.abs;

end
