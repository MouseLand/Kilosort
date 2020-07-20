function detset = setup_detector(rez, spkTh)

sig = 10;
dNearActiveSite = 30;
ops = rez.ops;

[ycup, xcup] = meshgrid(ops.yup, ops.xup);

NrankPC = 6;

NchanNear = 10;
[iC, dist] = getClosestChannels2(ycup, xcup, rez.yc, rez.xc, NchanNear);

igood = dist(1,:)<dNearActiveSite;
iC = iC(:, igood);
dist = dist(:, igood);

ycup = ycup(igood);
xcup = xcup(igood);

NchanNearUp =  10*NchanNear;
[iC2, dist2] = getClosestChannels2(ycup, xcup, ycup, xcup, NchanNearUp);

nsizes = 5;
v2 = gpuArray.zeros(5, size(dist,2), 'single');
for k = 1:nsizes
    v2(k, :) = sum(exp( - 2 * dist.^2 / (sig * k)^2), 1);
end

NchanUp = size(iC,2);

Params = [ops.NT ops.Nchan ops.nt0 NchanNear NrankPC ops.nt0min spkTh NchanUp NchanNearUp sig];

detset.Params = Params;
detset.iC = iC;
detset.iC2 = iC2;
detset.dist = dist;
detset.v2 = v2;
detset.dist2 = dist2;
detset.ycup = ycup;
detset.xcup = xcup;

detset.d2d = ((xcup - xcup').^2 + (ycup - ycup').^2).^.5;