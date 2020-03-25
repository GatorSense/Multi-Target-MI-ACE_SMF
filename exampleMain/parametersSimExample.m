function [sim,train_params,test_params] = parametersSimExample()

%Simulation
sim.NumReps = 1;           %number of simulations to run
sim.t1 = 1;                %target 1 choice (column taken from E_truth)
sim.t2 = 3;                %target 2 choice (column taken from E_truth)
sim.bt = [2,4,5];        %background signatures (column taken from E_truth)
sim.rand_seed_0 = 1;       %set random seed to 0 before first data generation
sim.train_on_test = 0;     %perform testing on training data instead

%Data Set Generation
load('E_truth.mat');

%Plot spectra from ground truth
plotSpectra(E_truth);

%Training Data Generation
train_params(1).E_target     = [sim.t1, sim.t2];    % index of target endmember
train_params(1).E_minus      = sim.bt;  % index of background endmembers
train_params(1).num_pbags    = 100; %per target
train_params(1).num_nbags    = 100; %per target
train_params(1).num_points   = 20; %per bag
train_params(1).n_tar        = 4;
train_params(1).N_b          = 1;
train_params(1).Pt_mean      = .2;
train_params(1).sigma        = 3;
train_params(1).expect_SdB   = 40;
train_params(1).bagData      = 1; 

%Testing Data Generation
test_params(1).E_target     = [sim.t1, sim.t2];    % index of target endmember
test_params(1).E_minus      = sim.bt;  % index of background endmembers
test_params(1).num_pbags    = 200; %per target
test_params(1).num_nbags    = 200; %per target
test_params(1).num_points   = 250; %per bag
test_params(1).n_tar        = 250;
test_params(1).N_b          = 1;
test_params(1).Pt_mean      = .1;
test_params(1).sigma        = 10;
test_params(1).expect_SdB   = 30;
test_params(1).bagData      = 1; 
end