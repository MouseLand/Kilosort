function [mh, imax, h1, h2] = split_histograms(his)

%% smooth the histogram
Nbins = size(his,1);

his2 = his;
his2(end, :) = 0;

% how much smoothing here matters! (adaptive?)
his2 = my_conv2(his2,1, 1);

Nfilt = size(his2,2);
his2max = zeros(Nbins, Nfilt, 2, 'single');

for j = 2:Nbins
    his2max(j, :,1)  = max(his2max(j-1,:,1), his2(j, :));    
end
for j = Nbins-1:-1:1
    his2max(j, :,2)  = max(his2max(j+1,:,2), his2(j, :));    
end
his2max = min(his2max, [], 3);

dh = his2max - his2;

hmax = zeros(100, Nfilt, 'single');
hwid = zeros(1, Nfilt, 'single');
for j = 1:Nbins
    hwid(dh(j,:)<1e-5) = 0;
    hwid =  hwid + dh(j,:);
    hmax(j, :) = hwid; 
end
hwid = zeros(1, Nfilt, 'single');
for j = Nbins:-1:1
    hmax(j, :) = hmax(j, :) + hwid; 
    hwid(dh(j,:)<1e-5) = 0;    
    hwid =  hwid + dh(j,:);    
end

dharea = hmax./sum(his2,1);

mh = max(dharea, [], 1);

[~, imax] = max((mh - dharea<1e-3) .* dh, [], 1);


h1 = zeros(size(his));
h2 = zeros(size(his));
for j = 1:Nbins
    h1(j, j<imax) = his(j, j<imax);
    h2(j, j>=imax) = his(j, j>=imax);
end

