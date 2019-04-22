%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         P_DECODE          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fs=p_decode(coded_filename,decoded_filename)

% birdie reduction constant (may add echo if too high)
rampConstant = 0.5; % [0.0 - 1.0]

% Read file header
fid = fopen(coded_filename,'r');
Fs          = fread(fid,1,'ubit16'); % Sampling Frequency
framelength = fread(fid,1,'ubit12'); % Frame Length
bitrate     = fread(fid,1,'ubit18'); % Bit Rate
scalebits   = fread(fid,1,'ubit4' ); % Number of Scale Bits per Sub-Band
num_frames  = fread(fid,1,'ubit26'); % Number of frames

prevInputValues = zeros(1,framelength/2);
for frame_count=1:num_frames
    
    % Read file contents
    qbits = sprintf('ubit%i', scalebits);
    gain = fread(fid,25,qbits);
    bit_alloc = fread(fid,25,'ubit4');
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
        if bit_alloc(ii) > 0
            qbits = sprintf('ubit%i', bit_alloc(ii)); 
            InputValues(indices(1):indices(end)) = fread(fid, length(indices) ,qbits);
        else
            InputValues(indices(1):indices(end)) = 0;
        end
    end

    % Dequantize values
    for ii=1:length(InputValues)
        if InputValues(ii) ~= 0
            if max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0) ~= 0
                InputValues(ii) = midtread_dequantizer(InputValues(ii),...
                    max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0));
            end
        end
    end

    gain2 = zeros(size(gain));
    for ii=1:25
        gain2(ii) = 2^gain(ii);
    end

    % Apply gain
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
        InputValues(indices(1):indices(end)) = InputValues(indices(1):indices(end)) * gain2(ii);
     end

    % Apply birdie reduction
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1    
        if bit_alloc(ii)<1
            InputValues(indices(1):indices(end)) = prevInputValues(indices(1):indices(end)) * rampConstant;
        end
    end
    
    % save this frame
    prevInputValues = InputValues;
    
    % Inverse MDCT
    x2((frame_count-1)*framelength+1:frame_count*framelength) = imdct(InputValues(1:framelength/2));
end

fclose(fid);
% Recombine frames
x3 = zeros(1,ceil((length(x2)-1)/2+1));
for ii=0:0.5:floor(length(x2)/(2*framelength))-1
    x3(ii*framelength+1 : (ii+1)*framelength) = x3(ii*framelength+1 : (ii+1)*framelength) + x2((2*ii)*framelength+1 : (2*ii+1)*framelength);
end

% Write file
audiowrite(decoded_filename,x3,Fs);