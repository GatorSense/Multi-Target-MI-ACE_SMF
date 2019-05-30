function [parameters] = setParameters()

%MTMIACE Parameters
parameters.methodFlag = 1;                  %ACE = 1, SMF = 0
parameters.numTargets = 2;                  %Num initialized targets
parameters.initType = 2;                    %Initialization method. (1) searches all positive instances and greedily selects instances that maximize objective. (2) K-Means clusters positive instances and greedily selects cluster centers that maximize objective.
parameters.optimize = 1;                    %Optimize signatures or not
parameters.maxIter = 1000;                  %Max number of iterations for optimization
parameters.globalBackgroundFlag = 0;        %Use all the data or not (including positive bags) to compute background statistics
parameters.posLabel = 1;                    %Labels for positive bags
parameters.negLabel = 0;                    %Labels for negative bags
parameters.numClusters = 20;                %Num clusters used for (only affects if initType 2 used)
parameters.alpha = 1;                       %Weight the uniqueness term gets in objective function, set to 0 if you do not want to use the term (affects initialization and optimization)

end
