

function predData = predictData(rez, samps)

W = gather_try(rez.W);
U = gather_try(rez.U);

samps = samps(1):samps(end);
buff = size(W,1); % width of a spike in samples

predData = zeros(size(U,1),numel(samps)+buff*4);

spikeTimes = rez.st3(:,1);


inclSpikes = spikeTimes>samps(1)-buff/2 & spikeTimes<samps(end)+buff/2;
st = spikeTimes(inclSpikes); % in samples

inclTemps = uint32(rez.st3(inclSpikes,2));
amplitudes = rez.st3(inclSpikes,3);

for s = 1:sum(inclSpikes)
    
    theseSamps = st(s)+(1:buff)-19-samps(1)+buff*2;
    predData(:,theseSamps) = predData(:,theseSamps) + ...
        squeeze(U(:,inclTemps(s),:)) * squeeze(W(:,inclTemps(s),:))' * amplitudes(s);
    
end

predData = predData(:,buff*2+1:end-buff*2).*rez.ops.scaleproc;