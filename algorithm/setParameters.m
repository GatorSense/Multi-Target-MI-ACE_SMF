function [parameters] = setParameters()

%General MTMIACE Parameters
parameters.methodFlag = 1;                  %ACE = 1, SMF = 0
parameters.numTargets = 2;                  %Num initialized targets
parameters.initType = 2;                    %Initialization method. (1) searches all positive instances and greedily selects instances that maximize objective. (2) K-Means clusters positive instances and greedily selects cluster centers that maximize objective.
parameters.optimize = 1;                    %Optimize signatures or not
parameters.maxIter = 1000;                  %Max number of iterations for optimization
parameters.globalBackgroundFlag = 0;        %Use all the data or not (including positive bags) to compute background statistics
parameters.posLabel = 1;                    %Labels for positive bags
parameters.negLabel = 0;                    %Labels for negative bags
parameters.numClusters = 20;                %Num clusters used for initType 2

%Uniqueness term settings (affects initialization and optimization)
parameters.uniqueTargets = 1;               %Use target uniqueness term or not
parameters.alpha = 1;                       %Weight the uniqueness term gets in objective function

%Initialization remove similar instances thesh. 
%Used in initType 1. If set to 1, only the initialized instance is excluded from being a future target signature. 
%Otherwise all instances that have a similarity more than this threshold to an initialized target are excluded from being a future target signature. Typical values around .9-.95 if used.
parameters.removeSimilarThresh = 1;         

end