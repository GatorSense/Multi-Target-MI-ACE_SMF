function results = optimizeTargets(data, initTargets, parameters)
%{
    Write ReadMe
%}

%% Set up data
nBags = length(data.dataBags);
nDim = size(data.dataBags{1}, 2);
nPBags = sum(data.labels == parameters.posLabel);

%Estimate background mean and inv cov
if(parameters.globalBackgroundFlag)
    dataBG = vertcat(data.dataBags{:});
    b_mu = mean(dataBG);
    b_cov = cov(dataBG)+eps*eye(size(dataBG, 2));
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
pDataBags = dataBagsWhitened(data.labels == parameters.posLabel);
nDataBags = dataBagsWhitened(data.labels == parameters.negLabel);

%Precompute term 2 in update equation
nMean = zeros(length(nDataBags), nDim);
for j = 1:length(nDataBags)
    nData = nDataBags{j};
    nMean(j, :) = mean(nData)';
end
nMean = mean(nMean);

%Initialize variables needed in while loop
numLearnedTargets = parameters.numTargets;
iter = 0;
continueFlag = 1;
optTargets = initTargets;
doneOptimizingFlags = zeros(1,numLearnedTargets);
targetIterationDoneCount = zeros(1,numLearnedTargets);
objTracker(parameters.maxIter) = struct();

%% Optimize target signatures
while(continueFlag && iter < parameters.maxIter)
    
    disp(['iter: ' num2str(iter)]);
    iter = iter + 1;

    optObjVal = zeros(numLearnedTargets,1);
    xStarsAll = cell(numLearnedTargets, nPBags);
    xStarsSimAll = zeros(numLearnedTargets, nPBags);
    pBagMaxIndex = zeros(numLearnedTargets, nPBags);
    
    %Compute xStars and xStarsSimilarity to know which signatures to include in optimization
    for target = 1:numLearnedTargets
        [optObjVal(target), xStarsAll(target,:), xStarsSimAll(target,:), pBagMaxIndex(target,:)] = evalObjectiveFunction(pDataBags, nDataBags, optTargets(target,:), optTargets, numLearnedTargets, parameters);
    end
    [maxXStarsSim, maxXStarsSimInd] = max(xStarsSimAll,[],1);
    
    %Track values across iterations
    if(~isnan(any(optObjVal)) && ~any(isnan(any(optTargets))))
        %Store the values at this iteration
        objTracker(iter).val = optObjVal;
        objTracker(iter).target = optTargets;
    else
        %There are objects that are nan, keep it the same as last iteration
        objTracker(iter).val = objTracker(iter-1).val;
        objTracker(iter).target = objTracker(iter-1).target;
    end
    objTracker(iter).pBagMaxIndex = pBagMaxIndex;
    objTracker(iter).numLearnedTargets = numLearnedTargets;
    
    %Update each target signature
    for target = 1:numLearnedTargets
        
        %Reinitialize pMean
        pMean = zeros(numLearnedTargets, size(optTargets,2));
        
        %If this target is not done optimizing, optimize it
        if(doneOptimizingFlags(target) == 0)
                            
            %If there are no xStars for this target, throw out the target signature
            if(sum(maxXStarsSimInd == target) == 0)
                numLearnedTargets = numLearnedTargets - 1;
                optTargets = [optTargets(1:target-1,:); optTargets(target+1:end,:)];
                objTracker(iter).val = [objTracker(iter).val(1:target-1); objTracker(iter).val(target+1:end)];
                objTracker(iter).target = [objTracker(iter).target(1:target-1,:); objTracker(iter).target(target+1:end,:)];
                break;
            else
                xStars = cell2mat(xStarsAll(target, maxXStarsSimInd == target));
                pMean(target,:) = mean(xStars, 2)';
            end

            %Update the target signature
            if(parameters.alpha ~= 0)             
                tMean = calcTargetMean(optTargets, target, parameters);
                t = pMean(target,:) - nMean - tMean;
            else
                t = pMean(target,:) - nMean;                
            end         
            
            %Avoid 0/0 error
            if(t ~= 0)
                optTargets(target, :) = t/norm(t);
            end
            
            %Set the targetIterationDoneCount if the target signatures to do not optimize to a single signature
            if(iter + 1 == parameters.maxIter)
                disp(['Stopping at MAX iter: ' num2str(parameters.maxIter)  ' for sig ' num2str(target) newline 'Unable to find optimal signature.']);
                targetIterationDoneCount(target) = iter+1;
            end    
            
        end
    end

    %If every target signature chooses the same positive bag representatives, then it is done optimizing.
    if(iter ~= 1)
        if(isequal(size(objTracker(iter-1).pBagMaxIndex), size(pBagMaxIndex)))
            if(~sum(objTracker(iter-1).pBagMaxIndex - pBagMaxIndex))
                continueFlag = 0;
                disp(['Stopping at iter: ', num2str(iter-1)]);
            end
        end
    end
end


%% Undo whitening
optTargets = (optTargets*D^(1/2)*V');
for tar = 1:numLearnedTargets
    optTargets(tar,:) = optTargets(tar,:)/norm(optTargets(tar,:));
end
initTargets = (initTargets*D^(1/2)*V');
for tar = 1:parameters.numTargets
    initTargets(tar,:) = initTargets(tar,:)/norm(initTargets(tar,:));
end

%Return results
results.optTargets = optTargets;
results.optObjVal = optObjVal;
results.b_mu = b_mu;
results.b_cov = b_cov;
results.sig_inv_half = sig_inv_half;
results.initTargets = initTargets;
results.methodFlag = parameters.methodFlag;
results.numTargets = numLearnedTargets;

end


%%
function [tMean] = calcTargetMean(tarSigs, currentTarInd, parameters)

otherTarSigs = tarSigs;
otherTarSigs(currentTarInd,:) = [];
tMean = mean(otherTarSigs, 1);

tMean = parameters.alpha * tMean;

end
