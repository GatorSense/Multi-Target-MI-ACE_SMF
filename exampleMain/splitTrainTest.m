function [ index_trainval, index_poly ] = splitTrainTest( metadata, classCol, nameCol )
%% Training/Validation
% This function randomly select polygons for 5 KFold cross validation.
%
% Parameters:
% 1) metadata: a cell array with all the metadata from the spectral library
% 2) classCol: integer designating what column of the metadata holds the
%    class name or dominant species name
% 3) nameCol: integer designating what column of the metadata hold the polygon ID  
% Returns:
% 1) index_trainval: an array with a size of metadata rows x iterations
%    that designates which indices are to be used in training (value of 1) and valdiation (0).
%    pixels not selected for training in a polygon will be set to a value of -1
% 2) index_poly: a cell array that contains a row for each pixel with the
%    polygon name
% ------------------------------------------------------------------------

% Set up variables
index_trainval = zeros(length(metadata), 1);  % an array
classList = unique(metadata(:, classCol));  % Determine the unique class, used to split into training/validation
index_poly = metadata(:,nameCol);  % Get names of ID for each pixel
    
% Pull out which polygons are used for training
for d = 1:size(classList,1)  % Loop through classes
    dominant_indices = find(strcmp(classList(d), metadata(:,classCol)));
    name = unique(metadata(dominant_indices,nameCol));  % Get the name of polygons for classes

    % Get training and validation separation
    indices = crossvalind('KFold', name, 5);

    % Loop through polygons and mark pixels
    for p = 1:size(name,1)
        index = find(strcmp(name(p), metadata(:,nameCol)) == 1);
        index_trainval(index) = indices(p);
    end
end
    
end



