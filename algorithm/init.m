classdef init
    %This class is for the initialization functions. There are two options
    %for initialzation of target signatures.
    % Functions included in this class definition:
    % nonOptTargets: Function that executes if non-optimization parameter is set. This does
    %                not perform optimization on initial targets using MT MI objective function.
    % init1: Initialize by searching all positive instances and greedily selects
    %        instances that maximizes objective function.
    % init2: Initialize by K-means cluster centers and greedily selecting cluster
    %        center that maximizes objective function.
    % removeSimilarData: removes data if it too similar to current target signature
    % computePDataSimilarityMatrix: computes similarity among positive bag pixels
    % computeNDataSimilarityMatrix: computes similarity among negative bag pixels 
    % --------------------------------------------------------------------
    
    methods(Static)
        function [results] = nonOptTargets(initTargets, parameters, dataInfo)
            % Function that executes if non-optimization parameter is set. This does
            % not perform optimization on initial targets using MT MI objective
            % function.
            % INPUTS:
            % 1) initTargets: matrix of initialized target signatures [n_targets, n_dim]
            %                 (will always be the number set in setParameters.m)
            % 2) parameters: a structure containing parameter variables. Parameters
            %                used in this function: numTargets, methodFlag
            % 3) dataInfo: background calculations (mu, inverse covariance)
            % OUTPUTS:
            % 1) results: a structure containing the following variables:
            %             1) b_mu: background mean [1, n_dim]
            %             2) b_cov: background covariance [n_dim, n_dim]
            %             3) sig_inv_half: inverse background covariance, [n_dim, n_dim]
            %             4) initTargets: the initial target signatures [n_targets, n_dim]
            %             5) methodFlag: value designating which method was used for similarity measure
            %             6) numTargets: the number of target signatures found
            %             7) optTargets: a string designating optimization was not performed
            % ------------------------------------------------------------------------
            
            % Set up Variables
            D = dataInfo.D;
            V = dataInfo.V;
            
            %Undo whitening
            initTargets = (initTargets*D^(1/2)*V');
            for tar = 1:parameters.numTargets
                initTargets(tar,:) = initTargets(tar,:)/norm(initTargets(tar,:));
            end
            
            % Save Variables
            results.b_mu = dataInfo.mu;
            results.b_cov = dataInfo.cov;
            results.sig_inv_half = dataInfo.invcov;
            results.initTargets = initTargets;
            results.methodFlag = parameters.methodFlag;
            results.numTargets = size(initTargets,1);
            results.optTargets = 'Optimization not performed, change settings in setParameters.m if desired';
            
        end
        
        function [initTargets, initTargetLocation, originalPDataBagNumbers, initObjectiveValue] = init1(pDataBags, nDataBags, parameters)
            % Initialize by searching all positive instances and greedily selects
            % instances that maximizes objective function.
            % INPUTS:
            % 1) pDataBags: a cell array of positive bags (already whitened)
            % 2) nDataBags: a cell array of negative bags (already whitened)
            % 3) parameters: a structure containing the parameter variables. variables
            %                used in this function - number of targets
            % OUTPUTS:
            % 1) initTargets: the initalized target signatures [n_targets, n_dims]
            % 2) initTargetLocation: the initalized target signatures index location[n_targets, 1]
            % 3) originalPDataBagNumbers: the number of original positive bags
            % 4) initObjectiveValue: the objective value from initalizations
            % -------------------------------------------------------------
            % Set up variables
            disp('Initializing Targets');
            pData = vertcat(pDataBags{:});
            initTargets = [];
            numPData = size(pData, 1);
            
            %Compute pDataConfidences
            [pDataConfidences, pDataBagNumbers] = computePDataSimilarityMatrix(pDataBags);
            originalPDataBagNumbers = pDataBagNumbers; %Keep a copy for labeling plots
            
            %Compute negative bag confidence values
            [nDataConfidences, nDataBagNumbers] = computeNDataSimilarityMatrix(pDataBags, nDataBags);
     
            %A boolean matrix to include the sample for a target concept consideration
            includeMatrix = ones(numPData, 1);
            initTargetLocation = zeros(2, parameters.numTargets);
            initObjectiveValue = zeros(1, parameters.numTargets);
            
            % Loop through all targets
            for target = 1:parameters.numTargets
                
                % Set up more variables for each target
                disp(['Initializing Target: ' num2str(target)]);
                objectiveValues = -100*ones(1, numPData);
                averagePBag = 100*ones(numPData, 1);
                averageNBag = 100*ones(numPData, 1);
                
                numTargetsLearned = target - 1;
                
                %After each target is calculated remove pData with that tag
                samplePBagMaxConfs = 100*ones(numPData, size(pDataBags, 2));
                
                %Compute objective function value for every pData sample
                for sampleNum = 1:numPData
                    %Only consider the samples that don't look like targets we've already chosen
                    if(includeMatrix(sampleNum))
                        [objectiveValues(sampleNum), ~, samplePBagMaxConfs(sampleNum,:), averagePBag(sampleNum), averageNBag(sampleNum)] = evalObjectiveFunctionLookup(numTargetsLearned, initTargetLocation, sampleNum, pDataBags, pDataConfidences, pDataBagNumbers, nDataConfidences, nDataBagNumbers, parameters);
                    end
                end
                
                %Take max objective value
                [initObjectiveValue(1,target), initTargetLocation(1,target)] = max(objectiveValues(:));
                
                %Store location and bag number for indexing in objective value function
                initTargetLocation(2,target) = originalPDataBagNumbers(initTargetLocation(1,target));
                
                %Extract sample at from the positive bags 
                initTarget = pData(initTargetLocation(1,target), :);
                initTarget = initTarget/norm(initTarget);
                
                % Store targets and the number of targets learned
                initTargets = vertcat(initTargets, initTarget);
                numTargetsLearned = target;
                
                %Remove similar data to target selected
                removeSimilarThresh = 1;
                [includeMatrix, pDataBagNumbers] = removeSimilarData(pData, pDataBagNumbers, initTargetLocation, numTargetsLearned, removeSimilarThresh); 
            end
        end
        
        function [initTargets, objectiveValues, C] = init2(pDataBags, nDataBags, parameters)
            % Initialize using K Means and picking cluster centers that maximize objective function.
            % INPUTS:
            % 1) pDataBags: a cell array of positive bags (already whitened)
            % 2) nDataBags: a cell array of negative bags (already whitened)
            % 3) parameters: a structure containing the parameter variables. variables
            %                used in this function - number of targets
            % OUTPUTS:
            % 1) initTargets: the initalized target signatures [n_targets, n_dims]
            % 2) objectiveValue: the objective value from initalizations
            % 3) C: the cluster centers from K-Means clustering [n_targets, n_dims]
            % -------------------------------------------------------------
            %Potential error handling
            if(parameters.numTargets > parameters.numClusters)
                msg = ['You must have more clusters than the number of targets set in the parameters' newline ...
                    blanks(5) 'Number of clusters (parameters): ' num2str(parameters.numClusters) newline ...
                    blanks(5) 'Number of targets (parameters): ' num2str(parameters.numTargets)];
                error(msg);
            end
            
            % Get Positive bag data
            pData = vertcat(pDataBags{:});
            
            % Get K-Means cluster centers (C)
            disp('Clustering Data');
            [~, C] = kmeans(pData, min(size(pData, 1), parameters.numClusters), 'MaxIter', parameters.maxIter);
            
            % Loop through targets
            initTargets = zeros(parameters.numTargets, size(C,2));
            numTargetsLearned = 0;
            for target = 1:parameters.numTargets
                % Set up variables for each target
                disp(['Initializing Target: ' num2str(target)]);
                objectiveValues = zeros(1, size(C,1));
                pBagMaxConf = zeros(size(C,1), size(pDataBags, 2));
                
                % Loop through cluster centers
                % Note for future: if large amount of data, can make this parfor loop
                for j = 1:size(C, 1) 
                    [objectiveValues(j), ~, pBagMaxConf(j,:)] = evalObjectiveFunction(pDataBags, nDataBags, C(j, :), initTargets, numTargetsLearned, parameters);
                end
                
                %Get location of max objective value
                [~, opt_loc] = max(objectiveValues);
                initTargets(target,:) = C(opt_loc, :);
                C(opt_loc,:) = [];
                
                numTargetsLearned = numTargetsLearned + 1;
            end
            
            %Normalize targets
            for target = 1:parameters.numTargets
                initTargets(target,:) = initTargets(target, :) / norm(initTargets(target, :));
            end
            
        end
        
        function [includeMatrix, pDataBagNumbers] = removeSimilarData(pData, pDataBagNumbers, initTargetLocation, numTargetsLearned, threshold)
            % Removes potential targets that look similar to the target already chosen
            % INPUTS:
            % 1) pData: a matrix of positive bag data
            % 2) pDataBagNumbers: the index locations of positive bag data
            % 3) initTargetLocation: the location of the initialized targets 
            % 4) numTargetsLearned: the number of targets that have been learned so far.
            % 5) threshold: If threshold = 1, only remove the one datapoint in the include matrix
            %               Else Remove datapoints that are similar to the targets already chosen
            % OUTPUTS
            % 1) includeMatrix: positive bag values that should be kept
            % 2) pDataBagNumbers: the number of pixels in a positive bag
            % ------------------------------------------------------------
            includeMatrix = ones(size(pData, 1), 1);
            
            % Loop through targets
            for target = 1:numTargetsLearned
                
                chosenSig = pData(initTargetLocation(target), :);
                similarity = sum(pData.*repmat(chosenSig, [size(pData, 1), 1]), 2);
                
                %Holds the indexes of pData that need to be excluded for learning the next target (ones should be included, zeros should be excluded)
                if(threshold == 1)
                    %If threshold = 1, only remove the one datapoint in the include matrix
                    includeMatrix(initTargetLocation(target), 1) = 0;
                else
                    %Remove datapoints that are similar to the targets already chosen
                    includeMatrix(similarity >= threshold) = 0;
                end
                
            end
            
            %Zero out pDataBagNumbers that correspond to the pData being removed for considerable targets (this is only computed for the most recently
            %learned target signature)
            if(threshold == 1)
                %If threshold = 1, only remove the one datapoint in the include matrix
                pDataBagNumbers(1,initTargetLocation(1,numTargetsLearned)) = 0;
            else
                %Remove datapoints that are similar to the targets already chosen
                pDataBagNumbers(similarity >= threshold) = 0;
            end
        end
        
        function [pDataConfidences, pDataBagNumbers] = computePDataSimilarityMatrix(pDataBags)
            % Computes objective function value for every pData sample
            % INPUTS:
            % 1) pDataBags: a cell array containing positive bags (already whitened)
            % OUTPUTS:
            % 1) pDataConfidences: the confidence value for each positive instance
            % 2) pDataBagNumbers: the number of pixels in each positive bag
            % ------------------------------------------------------------
            
            % Set up Variables
            pDataBagNumbers = [];
            numPBags = size(pDataBags, 2);
            allPData = vertcat(pDataBags{:});
            allPDataNumSamps = size(allPData,1);
            dataDimensions = size(pDataBags{1}, 2); %number of dimensions to the data
            
            % Loop through positive bags
            for dataBag = 1:numPBags
                
                % Store the dataBag number in a vector to be able to know what sample came from what bag in the pDataConfidences, needed for calculating
                % objective function value
                pBagNumSamps = size(pDataBags{dataBag}, 1);
                bagNumber = dataBag*ones(1, pBagNumSamps);
                pDataBagNumbers = horzcat(pDataBagNumbers, bagNumber);
            end
            
            %Preallocate pDataConfidences
            pDataConfidences = cell(1, numPBags);
            
            %Calculate the pDataConfidences for each dataBag
            for dataBag = 1:numPBags
                
                % Grab pixels in Individual positive bag
                pBag = pDataBags{dataBag};
                
                pBagNumSamps = size(pBag, 1);
                pBagReshape = reshape(pBag, [pBagNumSamps, 1, dataDimensions]);
                pBagCube = repmat(pBagReshape, 1, allPDataNumSamps);
                allPDataReshape = reshape(allPData, [1, allPDataNumSamps, dataDimensions]);
                allPDataCube = repmat(allPDataReshape, pBagNumSamps, 1);
                
                %Calculate confidences
                pBagConfidences = sum(pBagCube.*allPDataCube, 3);
                pDataConfidences{dataBag} = pBagConfidences;
            end    
        end
        
        function [nDataConfidences, nDataBagNumbers] = computeNDataSimilarityMatrix(pDataBags, nDataBags)
            % Computes the similarity between pixels
            % INPUTS:
            % 1) pDataBags: a cell array containing positive bags (already whitened)
            % 2) nDataBags: a cell array containing negative bags (already whitened)
            % OUTPUTS:
            % 1) nDataConfidences: the confidence value for each negative instance
            % 2) nDataBagNumbers: the number of pixels in each negative bag
            % ------------------------------------------------------------
            
            % Set up Variables
            allPData = vertcat(pDataBags{:});
            allNData = vertcat(nDataBags{:});
            nDataBagNumbers = [];
            pDataNumSamples = size(allPData,1);
            
            %Store the dataBag number in a vector to be able to know what sample came from waht bag in the nDataConfidences, needed for calculating
            %objective function value
            for dataBag = 1:size(nDataBags, 2)
                
                % Get number of pixels in negative bag
                numSamps = size(nDataBags{dataBag}, 1);
                
                %holds the bag number for all the samples in 'dataBag'
                bagNumber = dataBag*ones(1, numSamps);
                
                %Create a long vector that will be used to index nDataConfidences with the bag numbers
                nDataBagNumbers = horzcat(nDataBagNumbers, bagNumber);
                
            end
            
            %Preallocate nDataConfidences
            nDataConfidences = cell(1,size(nDataBags, 2));
            
            %Dimensionality of our data
            dataDimensions = size(pDataBags{1}, 2);
            
            %Calculate outside loop for efficiency. Does not depend on negative bag
            pDataReshape = reshape(allPData, [pDataNumSamples, 1, dataDimensions]);
            
            %Had to calculate this for each negative bag because the data was too large for matlab's array size limit
            for dataBag = 1:size(nDataBags, 2)
                
                nBagNumSamples = size(nDataBags{dataBag}, 1);
                
                %Individual positive bag
                nBag = nDataBags{dataBag};
                
                %reshape to do matrix wise calculation
                nBag = reshape(nBag, [1, nBagNumSamples, dataDimensions]);
                
                %Set up pDataCube for matrix calculation to be of size (numSamples in allPData x numSamples in nBag)
                pDataCube = repmat(pDataReshape, 1, nBagNumSamples);
                
                %Set up nBagCube for matrix calculation to be of size(
                nBagCube = repmat(nBag, pDataNumSamples, 1);
                
                %Calculate confidences at matrix level
                nBagNDataConfidences = sum(pDataCube.*nBagCube, 3);
                
                %Store each inside a cell list
                nDataConfidences{dataBag} = nBagNDataConfidences;
                
            end
        end
    end
end
