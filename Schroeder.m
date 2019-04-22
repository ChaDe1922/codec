%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          SCHROEDER        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function New_mask = Schroeder(Fs,N,freq,spl,downshift)
% Calculate the Schroeder masking spectrum for a given frequency and SPLkb
f_Hz = 1:Fs/N:Fs/2;                                                        %frequency in hertz equals an array from 1 to half the sampling rate (nyquist frequency), separated into segments Fs/N (sampling rate/ block size)

% Schroeder Spreading Function
dz = bark(freq)-bark(f_Hz);                                                % bark scale difference between the maskee (bark(freq)) and the masker (bark(f_Hz)) frequencies.
mask = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1 + (dz+0.474).^2);              % Schroeder spreading function equation

New_mask = (mask + spl - downshift);