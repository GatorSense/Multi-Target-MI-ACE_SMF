function [objVal, pConfMax] = evalObjectiveWhitenedMultiTar(pDataBags, nDataBags, targets, parameters)

pConfBags = zeros(length(pDataBags), 1);
tarMaxCount = zeros(size(targets, 1), 1);

for j = 1:length(pDataBags)
    pData = pDataBags{j};
    pConf = zeros(size(pData, 1), size(targets, 1));
    for k = 1:size(targets, 1)
        pConf(:, k) = sum(pData.*repmat(targets(k, :), [size(pData, 1), 1]), 2); %Confidence for kth target
    end
    if(parameters.abs)
        pConf = abs(pConf);
    end
    
    [maxConf, tarSelPerDP] = max(pConf(:));
    [selDP, tar] = ind2sub(size(pConf), tarSelPerDP);
    
    if(parameters.softmaxFlag) %Not updated for Multi target yet
        disp('Error: softmax not implemented.'); keyboard;
    else
        pConfBags(j) = maxConf; %Holds the max confidence score for every bag
        tarMaxCount(tar) = tarMaxCount(tar) + 1;
        pConfMax{tar}(tarMaxCount(tar), :) = pData(selDP, :)'; %Holds the target signatures with the max confidence score for every bag
    end
end

nConfBags = zeros(length(nDataBags), 2);
for j = 1:length(nDataBags)
    nData = nDataBags{j};
    nConf = zeros(size(nData, 1), size(targets, 1));
    for k = 1:size(targets, 1)
        nConf(:, k) = sum(nData.*repmat(targets(k, :), [size(nData, 1), 1]), 2);
        if(parameters.abs)
            nConf = abs(nConf);
        end
        for l = 1:size(nConf, 1)
            nConf = max(nConf, [], 2);
        end
    end
    nConfBags(j) = mean(nConf);
end

objVal = mean(pConfBags(:)) - mean(nConfBags(:)); %confidence that this dict looks like our desired target

end