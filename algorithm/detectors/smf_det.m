function [smf_data,smf_max,smf_idx,mu,siginv] = smf_det(hsi_data,tgt_sig,mu,siginv,meanFlag)
% Computes the Spectral Match Filter (SMF) detection statistic for each 
% sample in the data matrix given a set of targets.
% INPUTS:
% 1) hsi_data: input data for SMF detector, [n_dim, n_samples] 
% 2) tgt_sig: Target signature matrix, returned from miTargets.m, [n_dim, n_targets]
% 3) mu: background mean, returned from miTargets.m, [n_dim, 1]
% 4) siginv: background covariance, [n_dim, n_dim]
% 5) meanFlag: flag that indicates whether the background mean should be
%              subtracted from target signatures. Signatures from MT MI ACE
%              already have background mean substracted. 
%              0: do NOT remove background mean from target signatures
%              1: do remove background mean from target signatures
% OUTPUTS:
% 1) smf_data: smf confidence values for each sample [n_targets, n_samp] 
% 2) smf_max: max smf confidence value for each sample [1 x n_samp]
% 3) smf_idx: target signature index corresponding to
%             max smf confidence value for each sample. [1 x n_samp]
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
smf_data = zeros(size(tgt_sig,2),size(hsi_data,2));
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
    
    smf_data(i,:) = A./(B);
end

% Fuse SMF results from each dictionary element into a single confidence map.
[smf_max, smf_idx] = max(smf_data, [], 1);
end