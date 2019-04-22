function y = myNoiseGen(num_samples, w, noise_type)

delta = 1/(2^(w -1));

if strcmpi(noise_type,'rect')
    y = -delta/2 + rand(num_samples,1) * delta;
elseif strcmpi(noise_type,'tri')
    y = -delta/2 + rand(num_samples,1) * delta;
    y = y + (-delta/2 + rand(num_samples,1) * delta);
elseif strcmpi(noise_type,'hp')
    y = -delta/2 + rand(num_samples,1) * delta;
    y = filter([1 -1], 1, y);
else
    error('Unsupported noise type.');
end

end

