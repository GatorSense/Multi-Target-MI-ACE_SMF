function [ objectiveValue, pBagMaxConfSig, pBagMaxConf, averagePBag, averageNBag ] = evalObjectiveFunctionLookup( numTargetsLearned, targetIndexes,...
    sampleNum, pDataBags, pDataConfidences, pDataBagNumbers, nDataConfidences, nDataBagNumbers, parameters)
%EVALOBJECTIVEFUNCTIONLOOKUP

%Setup
allPData = vertcat(pDataBags{:});
numPBags = size(pDataBags, 2);
samplePBagMaxConf = zeros(1, numPBags);
pBagMaxConfSig = cell(1, numPBags);

%If we've already learned a target, compute the first part of the objective value for those targets
if(numTargetsLearned ~= 0)
    targets = targetIndexes(1, 1:numTargetsLearned);
    for bag = 1:numPBags
        %Check if the bag hasn't had all of it's data removed for potential targets
        if(sum(pDataBagNumbers == bag) ~= 0)
            targetPBagMaxConf(:, bag) = max(pDataConfidences{bag}(:,targets), [], 1); %Each row is a separate target
        else 
            targetPBagMaxConf(:, bag) = ones(size(targets,2),1);
        end
    end

%No targets learned yet, set to -one for when we take the max, we don't want this to affect taking the max confidence
else
    for bag = 1:numPBags
        targetPBagMaxConf(1, bag) = -1;
    end
end

%Get highest confidence per postive bag
for bag = 1:numPBags

    %Check if the bag hasn't had all of it's data removed for potential targets
    if(sum(pDataBagNumbers == bag) ~= 0)
        %Get max confidence of this bag
        [samplePBagMaxConf(1, bag), pBagMaxConfIndex] = max(pDataConfidences{bag}(:, sampleNum)); %Holds the max confidence score for every bag and stores the signatures in pBagMaxConfSig

        %Get actual signature of highest confidence in this bag
        pBagMaxConfSig{bag} = allPData(pBagMaxConfIndex, :)';

    %Place holder, set confidence to 0 for this bag, since they have all been removed from the potential targets. This means a previous target has already handled the data in this bag\
    %This will not negatively affect the overall result as it will take the max from the previously learned targets
    else
        samplePBagMaxConf(1, bag) = 0;
        pBagMaxConfSig{bag} = 1;
    end

end

%Take maximum confidence between this datapoint (sampleNum) and all other target signatures already learned 'greedy approach'
pBagMaxConf = vertcat(targetPBagMaxConf, samplePBagMaxConf);
pBagMaxConf = max(pBagMaxConf, [], 1);


%Get average confidence of each negative bag
numNBags = max(nDataBagNumbers);
nBagMeanConf = zeros(1, numNBags);

%If we've already learned a target, compute the second part of the objective value for those targets
if(numTargetsLearned ~= 0)
    targets = targetIndexes(1, 1:numTargetsLearned);
    for bag = 1:numNBags
        targetNBagConf = nDataConfidences{bag}(targets,:); %Get confidences associated with already learned target sigs
        sampleNBagConf = nDataConfidences{bag}(sampleNum,:); %Get confidences associated with sampleNum 'potential target sig'
        nBagConfs = vertcat(targetNBagConf, sampleNBagConf); %Concatenate confidences
        nBagConfs = max(nBagConfs, [], 1); %Take max of these confidences

        nBagMeanConf(:, bag) = mean(nBagConfs(:)); %Take average of the confidences for this bag
    end

%No targets learned yet, set to -1 for when we take the max, we don't want this to affect taking the max confidence
else
    for bag = 1:numPBags
        targetNBagConf(1, bag) = -1;
    end
end

%Get max of confidence for negative bag
%     for bag = 1:numNBags
%         nBagMeanConf(bag) = mean(nDataConfidences{bag}(sampleNum, :)); %Holds the average confidence score of the test sample against all samples in a negative bag
%     end

%Calculate actual objectiveValue
averagePBag = mean(pBagMaxConf(:));
averageNBag = mean(nBagMeanConf(:));

%Calculate uniqueness term
if(parameters.alpha ~= 0)
    if(numTargetsLearned ~= 0)
        targets = targetIndexes(:, 1:numTargetsLearned);
        uniqueness = calculateUniquenessTerm(pDataConfidences, sampleNum, targets, parameters);

    else
        uniqueness = 0;
    end

    objectiveValue = averagePBag - averageNBag - uniqueness;

else
    objectiveValue = averagePBag - averageNBag;
end


end

function [uniqueness] = calculateUniquenessTerm(pDataConfidences, testTargetIndex, targetsIndexes, parameters)

%Iterate over target indexes and 
numTargets = size(targetsIndexes,2);

similarities = zeros(1,numTargets);

for i = 1:numTargets
    targetBagNumber = targetsIndexes(2,i);
    
    similarities(1,i) = pDataConfidences{targetBagNumber}(1,testTargetIndex);
end

averageSim = mean(similarities);

uniqueness = parameters.alpha * averageSim;

end

