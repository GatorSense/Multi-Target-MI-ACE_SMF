function [data, labels, classes] = bagHyperspectral(spec, meta, colclass, colname, poslabel)
% Bag hyperspectral data into positive and negative bags based on separate
% metadata files. Metadata matrix is set up to contain data not used for
% bagging and the colclass, colname, and poslabel are used to determine how
% data will be bagged. 
% INPUTS:
% 1) spec: matrix containing spectra, size is samples x bands
% 2) meta: cell array containing metadata, size is samples x meta columns
% 3) colclass: an integer specifying which column of the metadata should be
%              used to group data into classes. Allows for multiple
%              classes instead of two. 
% 4) colname: an integer specifying which column of the metadata should be
%             used to group data into bags.
% 5) poslabel: (OPTIONAL) the label for the positive class and all other
%              classes will be set to negative bags. If not provided, 
%              the code will loop through classes and set each to a positive 
%              class while all other classes are negative. Results in labels 
%              variable with a column for each class.   
% OUTPUTS:
% 1) data: cell list with positive and negative bags (1xNumBags). 
%          Each cell contains a single bag in the form a (numInstances x instanceDimensionality) matrix 
% 2) labels: labels for dataBags [1, n_bags] OR [n_classes, n_bags]
%          * the labels should be a row vector with labels corresponding to the 
%          * parameters.posLabel and parameters.negLabel where a posLabel corresponds
%          * to a positive bag and a negLabel corresponds to a negative bag.
%          * The index of the label should match the index of the bag in dataBags
% 3) classes: a cell array with classes 
% -------------------------------------------------------------------------

% Set up Variables
classes = unique(meta(:,colclass));
polys = unique(meta(:,colname));
polys_class = cell(size(polys,1),1);

% Bag data
for i = 1:size(polys,1)
    idx = strmatch(polys(i), meta(:,colname));
    data.dataBags{i} = spec(idx,:); 
    polys_class(i) = unique(meta(idx,colclass));
end

% Set up corresponding labels
if nargin == 5 % If only one class is used
    labels = zeros(size(polys_class,1),1);
    labels(strcmp(polys_class,poslabel)) = 1;
    data.labels = labels;
else % If multiple classes are used
    labels = zeros(size(polys_class,1), size(classes,1));
    for i = 1:size(classes,1)
        idx = strmatch(classes(i), polys_class);
        labels(idx,i) = 1;
    end
end

end

