function [ace_data,ace_max,ace_idx,mu,siginv] = ace_det(hsi_data,tgt_sig,mu,siginv,meanFlag)
% Computes the Adaptive Cosine Estimator (ACE) detection statistic for each 
% sample in the data matrix given a set of targets.
% INPUTS:
% 1) hsi_data: input data for SMF detector, [n_dim, n_samples] 
% 2) tgt_sig: Target signature matrix, returned from miTargets.m, [n_dim, n_targets]
% 3) mu: background mean, returned from miTargets.m, [n_dim, 1]
% 4) siginv: background covariance, [n_dim, n_dim]
% 5) meanFlag: flag that indicates whether the background mean should be
%              subtracted from target signatures. Signatures from MT MI ACE
%              already have background mean subtracted. 
%              0: do NOT remove background mean from target signatures
%              1: do remove background mean from target signatures
% OUTPUTS:
% 1) ace_data: ace confidence values for each sample [n_targets, n_samp] 
% 2) ace_max: max ace confidence value for each sample [1 x n_samp]
% 3) ace_idx: target signature index corresponding to
%             max ace confidence value for each sample. [1 x n_samp]
% 4) mu: background mean, returned from miTargets.m, [n_dim, 1]
% 5) siginv: background covariance, [n_dim, n_dim]
% -----------------------------------------------------------------------

% Calculate mean and covariance from input data if it wasn't provided
if isempty(mu)
    mu = mean(hsi_data,2);
end
if isempty(siginv)
    siginv = pinv(cov(hsi_data'));
end

% Subtract background mean from input data
z = bsxfun(@minus,hsi_data,mu);

% Loop through targets 
ace_data = zeros(size(tgt_sig,2),size(hsi_data,2));
for i = 1:size(tgt_sig,2)
    
    if meanFlag == 1
        s = tgt_sig(:,i) - mu; %For target signatures pulled from data
    else
        s = tgt_sig(:,i); %For target signatures with mean removed
    end
    
    st_siginv = s'*siginv;
    st_siginv_s = s'*siginv*s;
    
    A = sum(st_siginv*z,1);
    B = sqrt(st_siginv_s);
    C = sqrt(sum(z.*(siginv*z),1));
    
    ace_data(i,:) = A./(B.*C);
end

% Fuse ACE results from each dictionary element into a single confidence map.
[ace_max, ace_idx] = max(ace_data, [], 1);
end