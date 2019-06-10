function [x1,y, phase] = getFreqBandEnv(x,fs,f_lo,f_hi,meth)
% purpose: extract frequency band envelope from a single EEG channel
% method:  (a) frequency band is computed using a zero-phase band-pass filter, 
%           (6th order Butterworth), envelope from the analytical signal
%           approximation (Hilbert transform)
%          (b) frequency band is computed using a continuous wavelet transform 
%           (Morlet, k0=6), envelope from the CWT-modulus, CWT implementation 
%           according to...
%           REF.: C. Torrence and G.P. Compo, A practical guide to wavelet
%           analysis, Bull. Am. Meteor. Soc. 79:61-78 (1998).
%
% rhs variables:
%               x: single row/column data vector
%               fs: sampling frequency [Hz]
%               f_lo: lower limit of alpha-band
%               f_hi: upper limit of alpha-band
%               meth: string, either 'bp' or 'cwt' (cf. method)
% lhs variable: 
%               y: frequency band envelope
%
% Author: Fred, 01/2009

% anti-alias filter coefficients
%[aa_b,aa_a]=butter(6,0.9);

y=[];
if strcmp(meth,'bp')
    % data properties
    [m,n] = size(x);
    if (m==1), x=x(:); end
    
    % get alpha band
    x1=getFreqBand(x,fs,f_lo,f_hi,meth);
    % get envelope
    y=abs(hilbert(x1));
    phase = angle(hilbert(x1));
    %y=filtfilt(aa_b,aa_a,y);
elseif (strcmp(meth,'cwt'))% meth=='cwt' (default)   
    
    % data properties
    ndata=length(x);
    [m,n] = size(x);
    if ((n>1) && (m>1))
        ndata=m; % time points, assuming 1 col./channel
    end
    if (m==1)
        ndata=n;
    end
        
    % cwt-Morlet default parameters
    dt=1/fs;
    T_lo = 1/f_hi;    T_hi = 1/f_lo;
    pad=0;            dj=0.1; % 0.25
    s0=T_lo;          J1=ceil(log2(T_hi/T_lo)/dj)
    mother='MORLET';  param=6;
    
    if ((n>1) && (m>1))
        xa=x;
        for i=1:n % filter each column (CWT)
            [cwt,period,scale,coi]=wavelet(x(:,i),dt,pad,dj,s0,J1,mother,param);
            y(:,i)=abs(sum(cwt,1));
        end
    else
    %if (m==1)
        [cwt,period,scale,coi]=wavelet(x(:),dt,pad,dj,s0,J1,mother,param);
        y=abs(sum(cwt,1));
        y=y(:);
    end
elseif (strcmp(meth,'spline'))% meth=='spline' (default)   
    
    % data properties
    ndata=length(x);
    [m,n] = size(x);
    if ((n>1) && (m>1))
        ndata=m; % time points, assuming 1 col./channel
    end
    if (m==1)
        ndata=n;
    end
    %n=length(x);
    dt=1/fs;
    
    % get band-pass
    xbp=getFreqBand(x,fs,f_lo,f_hi,'bp');
    
    if ((m>1) && (n>1))
        y=xbp;
        % filter each column
        for i=1:n
            y(:,i) = getPIPEnv(xbp(:,i));
        end
    %end
    else
    %if (m==1)
        y = getPIPEnv(xbp);
    end
else
    y=[];
end
    
end