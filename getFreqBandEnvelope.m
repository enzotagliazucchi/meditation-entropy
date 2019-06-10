function z = getFreqBandEnvelope(x,fs,f_lo,f_hi)
% purpose: extract a frequency band envelope
% method:  zero-phase band-pass filter (6th order Butterworth), Hilbert
% transform modulus
%
% input:
%               x: data as column(s)
%               fs: sampling frequency of x [Hz]
%               f_lo: lower limit of frequency band [Hz]
%               f_hi: upper limit of frequency band [Hz]
% output: 
%               z: frequency band envelope
%
% example: z = getFreqBandEnvelope(EEGdata,128,8,12)
% returns the alpha band (8-12 Hz) envelope of the 'EEGdata' signal,
% ('EEGdata' sampled at 128 Hz)
%
% Author: Fred, 01/2009

% frequency band extraction: zero-phase band-pass filtering
fNy=fs/2;      % Nyquist freq.
b_lo=f_lo/fNy; % normalized freq. [0..1]
b_hi=f_hi/fNy; % normalized freq. [0..1]

% filter coefficients (6th order Butterworth)
[bp_b1,bp_a1]=butter(6,b_hi);
[bp_b2,bp_a2]=butter(6,b_lo,'High');

% filter
[m,n] = size(x);
if ((n>1) && (m>1))
    y=x;
    % filter each column
    for i=1:n
       y(:,i) = filtfilt(bp_b1,bp_a1,x(:,i));  % zero-phase low-pass
       y(:,i) = filtfilt(bp_b2,bp_a2,y(:,i)); % zero-phase high-pass
    end
end
if ((m==1) || (n==1))
    x=x(:); % row-->col
    y=filtfilt(bp_b1,bp_a1,x);  % zero-phase low-pass
    y=filtfilt(bp_b2,bp_a2,y); % zero-phase high-pass
    y=y(:); % return as column vector
end

% envelope
z=abs(hilbert(y));