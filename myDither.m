function  y = myDither( x, w, noise_type) 

noise = myNoiseGen(size(x,1), w, noise_type);
x = x + noise;
y = midtread_quantizer(x, w);

end

