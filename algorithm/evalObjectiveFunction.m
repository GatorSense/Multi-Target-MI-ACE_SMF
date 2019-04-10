function [ objectiveValue, pBagMaxConfSig, pBagMaxConf ] = evalObjectiveFunction(pDataBags, nDataBags, targetSignature, targets, numTargetsLearned, parameters)
%N_EVALOBJECTIVEFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    %Setup
    numPBags = size(pDataBags, 2);
    pBagMaxConf = zeros(1, numPBags);
    pBagMaxConfSig = cell(1, numPBags);

    %Get highest confidence per postive bag
    for bag = 1:numPBags
    
        %Get data from specific bag
        pData = pDataBags{bag};
        
        %Confidences (dot product) of a sample across all other samples in pData, data has already been whitened
        pConf = sum(pData.*targetSignature, 2);
        
        %Get max confidence for this bag
        [pBagMaxConf(bag), pBagMaxConfIndex] = max(pConf);
        
        %Get actual signature of highest confidence in this bag
        pBagMaxConfSig{bag} = pData(pBagMaxConfIndex, :)';
    end
    
    
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
    if(parameters.uniqueTargets)
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

