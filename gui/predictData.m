

function predData = predictData(rez, samps)

W = gather_try(rez.W);
U = gather_try(rez.U);

samps = samps(1):samps(end);
buff = size(W,1); % width of a spike in samples

predData = zeros(size(U,1),numel(samps)+buff*4);

spikeTimes = rez.st3(:,1); % use the original spikes st2, not the merged and split


inclSpikes = spikeTimes>samps(1)-buff/2 & spikeTimes<samps(end)+buff/2;
st = spikeTimes(inclSpikes); % in samples

inclTemps = uint32(rez.st3(inclSpikes,2)); % use the original spikes st2, not the merged and split
amplitudes = rez.st3(inclSpikes,3);

for s = 1:sum(inclSpikes)
    ampi = amplitudes(s);
    Wi = sq(W(:, inclTemps(s), :));
    Ui = sq(U(:, inclTemps(s), :));
    
    % this is the reconstruction of the spatial part
    
    theseSamps = st(s)+(1:buff)-19-samps(1)+buff*2;
    predData(:,theseSamps) = predData(:,theseSamps) + Ui * Wi' * ampi;
    
    %     predData(:,theseSamps) = predData(:,theseSamps) + ...
    %         squeeze(U(:,inclTemps(s),:)) * squeeze(W(:,inclTemps(s),:))' * amplitudes(s);        
end

predData = predData(:,buff*2+1:end-buff*2).*rez.ops.scaleproc;