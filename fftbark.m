function b=fftbark(bin,N,Fs)
% b=fftbark(bin,N,Fs)
% Converts fft bin number to bark scale
% N is the fft length
% Fs is the sampling frequency
f = bin*(Fs/2)/N;
b = 13*atan(0.76*f/1000) + 3.5*atan((f/7500).^2); 