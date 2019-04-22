%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        P_ENCODE           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Quantized_Gain,quantized_words]=p_encode(x2,Fs,framelength,bit_alloc,~)

N = floor(fftbark(framelength/2,framelength/2,Fs))+1;
Gain = ones(N,1);
Quantized_Gain = zeros(N,1);

for ii=1:N
    indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
    Gain(ii) = 2^(ceil(log2((max(abs(x2(indices(1):indices(end))+1e-10))))));
    if Gain(ii) < 1
        Gain(ii) = 1;
    end
    x2(indices(1):indices(end)) = x2(indices(1):indices(end)) / (Gain(ii)+1e-10);
    Quantized_Gain(ii) = log2(Gain(ii));
end
    

for ii=1:numel(x2)
    quantized_words(ii) = midtread_quantizer(x2(ii), max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0)+1e-10); % 03/20/03
end
