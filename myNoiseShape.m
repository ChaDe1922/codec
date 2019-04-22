function  y = myNoiseShape( x, w, noise_type)

d = myNoiseGen(size(x,1), w, noise_type);
xq = midtread_quantizer(x+d,w);
y = x + filter([1 -1], 1, xq-x);

end

