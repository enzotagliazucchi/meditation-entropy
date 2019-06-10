function [C_all,K,C] = get_metastability(data, low, hi)

% obtains the Kuramoto order parameters 

C_all = [];
n=0;
for channel = 2:min(size(data,1),63)
    n=n+1
    temp = data(channel,:);
    [x1,y, phase] = getFreqBandEnv(temp,256,low,hi,'bp');
    C_all(n,:) = exp(i*phase);
 

end


K = var(abs(mean(C_all)));
C = mean(abs(mean(C_all)));