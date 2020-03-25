clear all; clc; close all; close all hidden;

% This will stop in debug mode if an error occurs so that you can examine all data, save whatever, etc.
dbstop if error

[sim,targets,train_params,test_params] = parameters();

results = {};
algName = {};
runTime = {};
AUC = {};
table = [];

if sim.rand_seed_0
    rng(20);
end

for iter = 1:sim.NumReps
    
    disp(['Run ',num2str(iter),' of ',num2str(sim.NumReps)]);
    
    %Generate Testing Data
    test = gen_2tar_data(targets,test_params);
    
    %Generate Training Data
    if(sim.train_on_test)
        train = test;
    else
        train = gen_2tar_data(targets,train_params);
        train.X = train.X/max(sqrt(sum(train.X.^2,1)));
    end
    
    pDataBags = train.dataBags(train.labels == 1);
    nDataBags = train.dataBags(train.labels == 0);

    n_settings = 1;

    % Set Parameters
    mi_params = setParameters();

    %Setting to let MTMIACE know this is simulated data
    mi_params.simulatedData = 1;

    %Run Using new miTargets function
    timerVal = tic;
    results{iter,n_settings} = miTargets(train,mi_params);
    timeToRun = toc(timerVal);
%             algName{iter,n_settings} = multiMTMIACEExcelSettings{n_settings, 'plotTitle'}{1};
    if(mi_params.methodFlag)
        algName{iter,n_settings} = 'Multi-target MI-ACE';
    else
        algName{iter,n_settings} = 'Multi-target MI-SMF';
    end
    runTime{iter,n_settings} = timeToRun;


    ace_out = [];
    for i = 1:results{iter,n_settings}.numTargets
        if(mi_params.methodFlag)
            out = ace_det(test.X ,results{iter,n_settings}.optTargets(i,:)', results{iter,n_settings}.b_mu', results{iter,n_settings}.sig_inv_half'*results{iter,n_settings}.sig_inv_half)';
        else
            out = smf_det(test.X ,results{iter,n_settings}.optTargets(i,:)', results{iter,n_settings}.b_mu', results{iter,n_settings}.sig_inv_half'*results{iter,n_settings}.sig_inv_half)';
        end
        ace_out = horzcat(ace_out,out);
    end
    ace_out = max(ace_out,[],2);
    [results{iter,n_settings}.xx, results{iter,n_settings}.yy, ~, results{iter,n_settings}.auc] = perfcurve(test.labels_point,ace_out,1);
    AUC{iter,n_settings} = results{iter,n_settings}.auc;
    
    %Plot learned target concepts and ROC
    sig_roc(results,algName);
    
    %print Run Times
    printRunTimes(runTime);
    
    %print AUC
    printAUC(AUC);
end

disp('done');

function [] = sig_roc(results,algName)

spectraLineWidth = 1.5;
rocLineWidth = 1;

for iter = 1:size(results,1)
    rocFig = figure();
    sigsFig1 = figure();
    sigsFig2 = figure();
    labels1 = {};
    labels2 = {};
    labels3 = {};
    for i = 1:size(results,2)
        figure(sigsFig1);
        hold on;
        plot(results{iter,i}.optTargets(1,:), 'LineWidth', spectraLineWidth); 
        labels1{end+1} = [algName{iter,i} ' Target 1 Concept'];    
        
        figure(sigsFig2);
        hold on;
        
        if(size(results{iter,i}.optTargets,1) > 1)
            plot(results{iter,i}.optTargets(2,:), 'LineWidth', spectraLineWidth); 
            labels2{end+1} = [algName{iter,i} ' Target 2 Concept'];
        end
        
        figure(rocFig);
        hold on;
        plot(results{iter,i}.xx, results{iter,i}.yy,  'LineWidth', rocLineWidth); 
        labels3{end+1} = [algName{iter,i}];

    end
    figure(sigsFig1); 
    legend(labels1); 
    axis([0 212 -0.2 0.2]); 
    title('Learned Target Signature 1');
    ylabel('Reflectance');
    xlabel('Wavelength (\mum)');
    xticklabels({'.4','.6','.8','1','1.2','1.4','1.6', '1.8', '2', '2.2', '2.4'});
    axis square;
    
    figure(sigsFig2); 
    legend(labels2); 
    axis([0 212 -0.2 0.2]);
    title('Learned Target Signature 2');
    ylabel('Reflectance');
    xlabel('Wavelength (\mum)');
    xticklabels({'.4','.6','.8','1','1.2','1.4','1.6', '1.8', '2', '2.2', '2.4'});
    axis square;
    
    figure(rocFig); 
    legend(labels3, 'Location', 'southeast'); 
    axis([0 1 0 1]); 
    title('Receiver Operator Characteristic Curve');
    xlabel('Probability of False Alarm'); 
    ylabel('Probability of Detection');
    axis square;
end
end

function [] = printRunTimes(runTime)

mySTR = [];
for i = 1:size(runTime,2)
    mySTR = strcat(mySTR, num2str(runTime{1,i}), ', ');
end

disp('RunTime');
disp(mySTR);

end

function [] = printAUC(AUC)

mySTR = [];
for i = 1:size(AUC,2)
    mySTR = strcat(mySTR, num2str(AUC{1,i}), ', ');
end
disp('AUC');
disp(mySTR);

end