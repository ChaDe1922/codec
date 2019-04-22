%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MIDTREAD_DEQUANTIZER    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret_value] = midtread_dequantizer(x,R)

sign = (2 * (x < 2^(R-1))) - 1;
Q = 2 / (2^R - 1); 

x_uint = uint32(x);
x = bitset(x_uint,R,0);      
x = double(x);

ret_value = sign * Q .* x;