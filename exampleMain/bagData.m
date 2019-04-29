function [data] = bagData(yourData)
% You will need to edit this code to bag your data according to the structure of your data. This is provided to provide a template to how
% the data structure should be returned. How you construct the bags is dependent on your application.
%{
bagData() should return:
data.dataBags
data.labels

data:
    dataBags: bagged data
        * a positive bag should have at least one positive instance in it
        * a negative bag should consist of all negative instances

    labels: labels for dataBags
        * the labels should be a row vector with labels corresponding to the 
        * parameters.posLabel and parameters.negLabel where a posLabel corresponds
        * to a positive bag and a negLabel corresponds to a negative bag.
        * The index of the label should match the index of the bag in dataBags
%}

%Example bag dimensionality (numInstances x dimensionality) matrix. Has 5 instances, each with 100 dimensionality:
exampleBag = rand(5,100);

%Set up data structure
numBags = yourData.numPBags + yourData.numNBags; %total number of bags = number of positive bags + number of negative bags
data.dataBags = cell(1,numBags);
data.labels = zeros(1,numBags);

%Store positive bags and labels
for i = 1:numPBags
    %yourData.posBag(i) should be of the same format as the exampleBag (in a real application this matrix would come from your data)
    yourData.posBag(i) = exampleBag;
    
    %store positive bag in data structure
    data.dataBags{i} = yourData.posBag(i); 
    
    %set corresponding label to 1
    data.labels(i) = 1;
end

%Store negative bags and labels
for i = 1:numNBags
    %yourData.negBag(i) should be of the same format as the exampleBag (in a real application this matrix would come from your data)
    yourData.negBag(i) = exampleBag;
    
    %store negative bag in data structure
    data.dataBags{numPBags+i} = yourData.negBag(i);
    
    %set corresponding label to 0
    data.labels(numPBags+i) = 0;
end

end