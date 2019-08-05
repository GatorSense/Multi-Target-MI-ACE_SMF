function results = optimizeTargets(initTargets, pDataBags, nDataBags, parameters, dataInfo)
% Function that evaluates the objective function for multiple target 
% multiple instance ACE/SMF algorithm to optimize target signatures.
% INPUTS:
% 1) initTargets: the initalized target signatures [n_targets, n_dims]
% 2) pDataBags: a cell array containing the positive bags (already whitened)
% 3) nDataBags: a cell array containing the negative bags (already whitened)
% 4) parameters: a structure containing the parameter variables 
% 5) dataInfo: a structure that contains information about the background.
%              Not used in the function, but save variables out to results
%              1) mu: background mean [1, n_dim]
%              2) cov: background covariance [n_dim, n_dim]
%              2) invcov: inverse background covariance, [n_dim, n_dim]
%              3) D: a singular value decomposition of matrix A, such that A = U*D*V'.
%              4) V: a singular value decomposition of matrix A, such that A = U*D*V'.
%              5) U: a singular value decomposition of matrix A, such that A = U*D*V'.
% OUTPUTS:
% 1) results: a structure containing the following variables:
%             1) b_mu: background mean [1, n_dim]
%             2) b_cov: background covariance [n_dim, n_dim]
%             3) sig_inv_half: inverse background covariance, [n_dim, n_dim]
%             4) initTargets: the initial target signatures [n_targets, n_dim]
%             5) methodFlag: value designating which method was used for similarity measure
%             6) numTargets: the number of target signatures found
%             7) optTargets: the optimized target signatures [n_opttargets,
%                n_dim] Might have fewer targets then initTargets
% -------------------------------------------------------------------------
%% Set up data
nDim = size(pDataBags{1}, 2);
nPBags = size(pDataBags, 2);

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
D = dataInfo.D;
V = dataInfo.V;
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
results.b_mu = dataInfo.mu;
results.b_cov = dataInfo.cov;
results.sig_inv_half = dataInfo.invcov;
results.initTargets = initTargets;
results.methodFlag = parameters.methodFlag;
results.numTargets = numLearnedTargets;

end

function [tMean] = calcTargetMean(tarSigs, currentTarInd, parameters)
% Function that calculates the target mean
% INPUTS:
% 1) tarSigs: the target signatures [n_targets, n_dim]
% 2) currentTarInd: the current target index
% 3 parameters: structure containing parameter settings. Variables used are
%               alpha
% OUTPUTS:
% 1) tMean: the target signatures mean
% -----------------------------------------------------------------------
otherTarSigs = tarSigs;
otherTarSigs(currentTarInd,:) = [];
tMean = mean(otherTarSigs, 1);
tMean = parameters.alpha * tMean;

end
