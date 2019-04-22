clear; clc; 

ref = 'scary.wav'; % 16 times oversampling
test = 'yourfile_decoded.wav'; % 4 times oversampling

[odg, movb] = PQevalAudio_fn(ref, test);