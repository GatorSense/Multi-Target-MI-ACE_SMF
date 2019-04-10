function [sim,targets,train_params,test_params] = parameters()

%Simulation
sim.NumReps = 1;           %number of simulations to run
sim.t1 = 3;                %target 1 choice (column taken from E_truth)
sim.t2 = 4;                %target 2 choice (column taken from E_truth)
sim.rand_seed_0 = 1;       %set random seed to 0 before first data generation
sim.train_on_test = 0;     %perform testing on training data instead

%Data Set Generation
E_truth = 0;
load('E_truth.mat');
denom = sum(E_truth.*E_truth);
E_truth = E_truth./repmat(denom,[size(E_truth,1),1]);
targets.E_t1 = E_truth(:,sim.t1);
targets.E_t2 = E_truth(:,sim.t2);
targets.E_minus = E_truth(:,setdiff([1 2 3 4], [sim.t1 sim.t2]));

%Plot spectra from ground truth
plotSpectra(E_truth);

%Training Data Generation
train_params(1).num_pbags = 100; %per target
train_params(1).num_nbags = 100; %per target
train_params(1).num_points = 20; %per bag
train_params(1).n_tar = 4;
train_params(1).N_b = 1;
train_params(1).Pt_mean = .2;
train_params(1).sigma = 3;
train_params(1).expect_SdB = 40;

%Testing Data Generation
test_params(1).num_pbags = 200; %per target
test_params(1).num_nbags = 200; %per target
test_params(1).num_points = 250; %per bag
test_params(1).n_tar = 250;
test_params(1).N_b = 1;
test_params(1).Pt_mean = .1;
test_params(1).sigma = 10;
test_params(1).expect_SdB = 30;

end