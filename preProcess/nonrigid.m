function [disp,m] = nonrigid(F,N,S)
%NONRIGID Summary of this function goes here
%   Detailed explanation goes here

if nargin==1
    N = [128 64 32 16]; % iteration number for each level of pyramid
    S = 1.5; % smoothing
elseif nargin==2
    S = 1.5;
end

nt = size(F,3);
if(nt==1)
    m = F(:,:,1);
    disp = {zeros([size(m) 2])};
elseif (nt==2) % align first to second frame
    f1 = F(:,:,1);
    f2 = F(:,:,2);
    [d,f12] = imregdemons(f1,f2,N,'AccumulatedFieldSmoothing',S,'PyramidLevels',length(N),'DisplayWaitBar',false);
    m = (f12+f2)/2;
    disp = {d zeros([size(f1) 2])};
else
    idx1 = 1:floor(nt/2); % split frames in two
    idx2 = floor(nt/2)+1 : nt; % recursive alignment
    [d1,f1] = nonrigid(F(:,:,idx1));
    [d2,f2] = nonrigid(F(:,:,idx2));
    [d,f12] = imregdemons(f1,f2,N,'AccumulatedFieldSmoothing',S,'PyramidLevels',length(N),'DisplayWaitBar',false);
    m = (f12+f2)/2;
    d1 = cellfun(@(x)x+d,d1,'UniformOutput',false); % concatenate distortions
    disp = [d1 d2];
end

end

