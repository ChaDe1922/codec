%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    MIDTREAD_QUANTIZER     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret_value] = midtread_quantizer(x,R)

Q = 2 / (2^R - 1);      
q = quant(x,Q);
s = q<0;    
ret_value = (abs(q)./Q + s*2^(R-1)); %uint16 PUT THIS BACK