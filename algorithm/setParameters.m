function [parameters] = setParameters()
% Parameters function for Multiple Target Multiple Instance (MT MI)
% Adaptive Cosine Estimator and Spectral Match Filter

parameters.methodFlag = 1;                  % Detemine similarity method 
                                            % 0: Spectral Match Filter (SMF)
                                            % 1: Adaptive Cosine Estimator (ACE)

parameters.numTargets = 2;                  % Set number of initialized targets

parameters.initType = 2;                    % Set Initialization method used to obtain initalized targets 
                                            % 1: searches all positive instances and greedily selects instances that maximize objective. 
                                            % 2: K-Means clusters positive instances and greedily selects cluster centers that maximize objective.

parameters.numClusters = 20;                % Number of clusters used for K-Means (only affects if initType 2 used)

parameters.optimize = 1;                    % Determine whether to optimize target signatures
                                            % 0: Do not optimize target signatures
                                            % 1: Optimize target signatures using MT MI
                                            
parameters.maxIter = 1000;                  % Max number of iterations for optimization

parameters.globalBackgroundFlag = 0;        % Determines how the background mean and inverse covariance are calcualted  
                                            % 0: mean and cov calculated from all data (positive and negative data)
                                            % 1: mean and cov calculated from only negative bags

parameters.posLabel = 1;                    % Labels for positive bags
parameters.negLabel = 0;                    % Labels for negative bags

parameters.alpha = 1;                       % Weight the uniqueness term gets in objective function
                                            % set to 0 if you do not want to use the term (affects initialization and optimization)

end
