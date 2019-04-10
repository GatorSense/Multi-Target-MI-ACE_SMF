%% Read Me:
% This script is a sample main which you can use to run MTMIACE. More detailed info in miTargets.m


%% Clear all 
close all; clear all; clc; fclose all;
disp('Running ...');

% This will stop in debug mode if an error occurs so that you can examine all data, save variables, etc.
dbstop if error


%% Bag data

disp('Bagging data');

%{
Bag your specific data here:

example-
[data] = bagData(rawData);

%}

%% Set parameters

disp('Setting parameters');

parameters = setParameters();


%% Run 

disp('Running MTMIACE');

%Uncomment below when you've bagged your data
%[results] = miTargets(data, parameters)


%% Visualize your results

disp('done');