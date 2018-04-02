function [data] = bagData(rawData)

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

%Bag your data here
data.dataBags = [];
data.labels = [];

end