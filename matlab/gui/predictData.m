

function predData = predictData(rez, samps)

W = gather_try(rez.W);
U = gather_try(rez.U);

samps = samps(1):samps(end);
buff = size(W,1); % width of a spike in samples

predData = zeros(size(U,1),numel(samps)+buff*4);

spikeTimes = rez.st2(:,1); % use the original spikes st2, not the merged and split


inclSpikes = spikeTimes>samps(1)-buff/2 & spikeTimes<samps(end)+buff/2;
st = spikeTimes(inclSpikes); % in samples

inclTemps = uint32(rez.st2(inclSpikes,2)); % use the original spikes st2, not the merged and split
amplitudes = rez.st2(inclSpikes,3);

for s = 1:sum(inclSpikes)
    ibatch = ceil(st(s)/rez.ops.NT); % this determines what batch the spike falls in
    ampi = rez.muA(inclTemps(s), ibatch);
    
    % this is the reconstruction of the temporal part
    Wi = rez.W_a(:, :, inclTemps(s)) * rez.W_b(ibatch, :, inclTemps(s))'; 
    Wi = reshape(Wi, rez.ops.nt0, []);
    
    % this is the reconstruction of the spatial part
    Ui = rez.U_a(:, :, inclTemps(s)) * rez.U_b(ibatch, :, inclTemps(s))';
    Ui = reshape(Ui, rez.ops.Nchan, []);
    
    theseSamps = st(s)+(1:buff)-19-samps(1)+buff*2;
    predData(:,theseSamps) = predData(:,theseSamps) + Ui * Wi' * ampi;
    
    %     predData(:,theseSamps) = predData(:,theseSamps) + ...
    %         squeeze(U(:,inclTemps(s),:)) * squeeze(W(:,inclTemps(s),:))' * amplitudes(s);        
end

predData = predData(:,buff*2+1:end-buff*2).*rez.ops.scaleproc;