function [dataBagsWhitened, dataInfo] = whitenData(data, parameters)
% Function that 'whitens' data for ACE and SMF detectors. Removes
% background mean and scales data.
% INPUTS:
% 1) data: bagged data 
% 2) parameters: structure containing parameter settings 
%                For details see setParameters.m. Variables used in this function:
%                methodFlag: Determines which method is used as similarity measure    
%                  0: Spectral Match Filter (SMF)
%                  1: Adaptive Cosine Estimator (ACE)
%                globalBackgroundFlag: Determines how the background mean and cov is calcualted 
%                  0: mean and cov calculated from all data 
%                  1: mean and cov calculated from only negative bags
%                negLabel: Labels for negative bags
% OUTPUTS:
% 1) dataBagsWhitened: input data that is whitened, same format and shape
%                      as input data
% 2) dataInfo: a structure that contains information about the background
%              1) mu: background mean [1, n_dim]
%              2) cov: background covariance [n_dim, n_dim]
%              2) invcov: inverse background covariance, [n_dim, n_dim]
%              3) D: a singular value decomposition of matrix A, such that A = U*S*V'.
%              4) V: a singular value decomposition of matrix A, such that A = U*S*V'.
%              5) U: a singular value decomposition of matrix A, such that A = U*S*V'.
% -----------------------------------------------------------------------

%Estimate background mean and inv cov
if(parameters.globalBackgroundFlag)
    dataBG = vertcat(data.dataBags{:});
    b_mu = mean(dataBG);
    b_cov = cov(dataBG)+eps*eye(size(dataBG, 2));
else
    nData = vertcat(data.dataBags{data.labels == parameters.negLabel});
    b_mu = mean(nData);
    b_cov = cov(nData)+eps*eye(size(nData, 2));

end

%Whiten Data
[U, D, V] = svd(b_cov);
sig_inv_half = D^(-1/2)*U';
dataWhitened = {};
nBags = length(data.dataBags);
nDim = size(data.dataBags{1}, 2);
for i = 1:nBags
    m_minus = data.dataBags{i} - repmat(b_mu, [size(data.dataBags{i}, 1), 1]);
    m_scale = m_minus*sig_inv_half';
    if(parameters.methodFlag)
        denom = sqrt(repmat(sum(m_scale.*m_scale, 2), [1, nDim]));
        dataWhitened{i} = m_scale./denom;
    else
        dataWhitened{i} = m_scale;
    end
end

dataBagsWhitened.dataBags = dataWhitened;
dataBagsWhitened.labels = data.labels;
dataInfo.mu = b_mu;
dataInfo.cov = b_cov;
dataInfo.invcov = sig_inv_half;
dataInfo.D = D;
dataInfo.V = V;
dataInfo.U = U;
end

