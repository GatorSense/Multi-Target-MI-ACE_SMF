function [parameters] = setParameters()

% Set MTMIACE Parameters
parameters.numTargets = 1;
parameters.initType = 4; 
parameters.optimize = 0;
parameters.maxIter = 100;
parameters.methodFlag = 1;
parameters.globalBackgroundFlag = 0;
parameters.posLabel = 1;
parameters.negLabel = 0;
parameters.abs = 0;
parameters.softmaxFlag = 0;
parameters.samplePor = 1;

end