function y=getFreqBand(x,fs,f_lo,f_hi,meth)

% method:  (a) zero-phase band-pass filter, 6th order Butterworth
%          (b) continuous wavelet transform (Morlet, k0=6), implementation
%           according to...
%           REF.: C. Torrence and G.P. Compo, A practical guide to wavelet
%           analysis, Bull. Am. Meteor. Soc. 79:61-78 (1998).
%
% rhs variables:
%               x: single row/column data vector
%               fs: sampling frequency [Hz]
%               f_lo: lower limit of frequency band
%               f_hi: upper limit of frequency band
%               meth: string, either 'bp' or 'cwt' (cf. method)
% lhs variable: 
%               x: frequency band
%

y=[];
if strcmp(meth,'bp')
    
    % zero-phase band-pass filtering
    fNy=fs/2;      % Nyquist freq.
    b_lo=f_lo/fNy; % normalized freq. [0..1]
    b_hi=f_hi/fNy; % normalized freq. [0..1]
    % filter coefficients
    [bp_b1,bp_a1]=butter(6,b_hi);
    [bp_b2,bp_a2]=butter(6,b_lo,'high');
    % filter
    [m,n] = size(x);
    if ((n>1) && (m>1))
        y=x;
        % filter each column
        for i=1:n
           y(:,i) = filtfilt(bp_b1,bp_a1,x(:,i));
           y(:,i) = filtfilt(bp_b2,bp_a2,y(:,i));
        end
    end
    if ((m==1) || (n==1))
        x=x(:); % row-->col
        y=filtfilt(bp_b1,bp_a1,x);  % zero-phase low-pass
        y=filtfilt(bp_b2,bp_a2,y); % zero-phase high-pass
        y=y(:); % return as column vector
    end
    
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
            y(:,i)=real(sum(cwt,1));
        end
    else
    %if (m==1)
        [cwt,period,scale,coi]=wavelet(x(:),dt,pad,dj,s0,J1,mother,param);
        y=real(sum(cwt,1));
        y=y(:);
    end
else
    y=[];
end
    
end