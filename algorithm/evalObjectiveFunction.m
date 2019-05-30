function [ objectiveValue, pBagMaxConfSig, pBagMaxConf] = evalObjectiveFunction(pDataBags, nDataBags, targetSignature, targets, numTargetsLearned, parameters)
%N_EVALOBJECTIVEFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    %Setup
    numPBags = size(pDataBags, 2);
    pBagMaxConfThisTarget = zeros(1, numPBags);
    pBagMaxConfAll = zeros(1, numPBags);
    pBagMaxConfSig = cell(1, numPBags);

    %Get the max confidence of the positive bag representatives for all targets
    for bag = 1:numPBags
       
        %Get data from specific bag
        pData = pDataBags{bag};
        
        %Resize bag data to compute similarity for all targets at once
        pDataMat = repmat(pData, size(targets,1), 1);
        
        %Resize targets to match number of elements in this bag
        targetsMat = repmat(targets, size(pData,1), 1);
        
        %Confidences (dot product) of a sample across all other samples in pData, data has already been whitened
        pConf = sum(pDataMat.*targetsMat, 2);
        
        %Get max confidence for this bag
        [pBagMaxConfAll(bag), pBagMaxConfIndex] = max(pConf);
    end
    
    
    %Get the max confidence of the positive bag representatives for this target
    for bag = 1:numPBags
    
        %Get data from specific bag
        pData = pDataBags{bag};
        
        %Confidences (dot product) of a sample across all other samples in pData, data has already been whitened
        pConf = sum(pData.*targetSignature, 2);
        
        %Get max confidence for this bag
        [pBagMaxConfThisTarget(bag), pBagMaxConfIndex] = max(pConf);
        
        %Get actual signature of highest confidence in this bag
        pBagMaxConfSig{bag} = pData(pBagMaxConfIndex, :)';
    end
    
    %Take max across this target and all targets
    pBagMaxConf = max([pBagMaxConfAll; pBagMaxConfThisTarget], [], 1);
    
    %Get average confidence of each negative bag
    numNBags = length(nDataBags);
    nBagMeanConf = zeros(1, numNBags);
    
    for bag = 1:numNBags

        %Get data from specific bag
        nData = nDataBags{bag};
        
        %Confidence (dot product) of a sample across all other samples in nData, data has already been whitened
        nBagMeanConf(bag) = mean(sum(nData.*targetSignature, 2));

    end
    
    %Calculate actual objectiveValue
    averagePBag = mean(pBagMaxConf(:));
    averageNBag = mean(nBagMeanConf(:));
    
    %Calculate uniqueness term
    if(parameters.alpha ~= 0)
        if(numTargetsLearned ~= 0)
            uniqueness = calculateUniquenessTerm(targetSignature, targets, parameters);
        else
            uniqueness = 0;
        end
        objectiveValue = averagePBag - averageNBag - uniqueness;
    else
        objectiveValue = averagePBag - averageNBag;
    end

end


%%
function [uniqueness] = calculateUniquenessTerm(targetSignature, targets, parameters)

similarities = sum(targets.*targetSignature, 2);

averageSim = mean(similarities);

uniqueness = parameters.alpha * averageSim;

end

