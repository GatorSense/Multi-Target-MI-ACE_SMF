function [smf_data,mu,siginv] = smf_det(hsi_data,tgt_sig,mu,siginv)

if isempty(mu)
    mu = mean(hsi_data,2);
end
if isempty(siginv)
    siginv = pinv(cov(hsi_data'));
end

%For target signatures pulled from data, use line 11 instead of 12
% s = tgt_sig - mu;
s = tgt_sig;
z = bsxfun(@minus,hsi_data,mu);

st_siginv = s'*siginv;
st_siginv_s = s'*siginv*s;

A = sum(st_siginv*z,1);
B = sqrt(st_siginv_s);
C = sqrt(sum(z.*(siginv*z),1));

smf_data = A./(B);

end