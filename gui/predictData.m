

function predData = predictData(rez, samps)

samps = samps(1):samps(end);
buff = size(rez.W,1); % width of a spike in samples

predData = zeros(size(rez.U,1),numel(samps)+buff*2);

spikeTimes = rez.st3(:,1);


inclSpikes = spikeTimes>samps(1)-buff/2 & spikeTimes<samps(end)+buff/2;
st = spikeTimes(inclSpikes); % in samples

inclTemps = uint32(rez.st3(inclSpikes,2));
amplitudes = rez.st3(inclSpikes,3);

for s = 1:sum(inclSpikes)
    
    theseSamps = st(s)+(1:buff)-ceil(buff/2)-samps(1)+buff+1;
    predData(:,theseSamps) = predData(:,theseSamps) + ...
        squeeze(rez.U(:,inclTemps(s),:)) * squeeze(rez.W(:,inclTemps(s),:))' * amplitudes(s);
    
end

predData = predData(:,buff+1:end-buff).*rez.ops.scaleproc;