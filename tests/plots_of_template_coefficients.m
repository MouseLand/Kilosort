testID = 156;
mu = rez.mu;
clusterIDs = rez.st3(:,2);
% tfi = rez.iNeigh;
% tf = rez.cProj;

spikesTest = find(clusterIDs==testID);

simIDs = rez.iNeigh(:,testID);
%
figure(2)
clf
figure(1)
clf

LAM = ops.lam(3) * (20./mu).^2;

nSP = ceil(sqrt(length(simIDs)));
for s = 1:length(simIDs)
    simS_T = find(rez.iNeigh(:,simIDs(s))==testID);
    spikesSim = find(clusterIDs==simIDs(s));
    
     if simIDs(s)~=testID && numel(spikesSim)>20 && ~isempty(simS_T)
        figure(2)
        subplot(4, 8, s);
        
        plot(rez.cProj(spikesTest,1), rez.cProj(spikesTest,s), '.')
        hold on;
        plot(rez.cProj(spikesSim,simS_T), rez.cProj(spikesSim,1), '.')
        
        title(sprintf('%d vs %d', testID, simIDs(s)))
        axis tight
        
        figure(1)
        subplot(nSP/2, 2*nSP, s);
         
        ft1 = [rez.cProj(spikesTest,1); rez.cProj(spikesSim,simS_T)];
        ft2 = [rez.cProj(spikesTest,s); rez.cProj(spikesSim,1)];
        
        ft1 = (ft1 + LAM(testID) * mu(testID))    / sqrt(1 + LAM(testID));
        ft2 = (ft2 + LAM(simIDs(s)) * mu(simIDs(s))) / sqrt(1 +  LAM(simIDs(s)));
        
        df = ft1 - ft2;
        l1 = min(df(:));
        l2 = max(df(:));
%         bins = linspace(l1, l2, 100);
        df1 = df(1:numel(spikesTest));
        df2 = df(1+numel(spikesTest):end);
        
        se = (std(df1) + std(df2))/2;
        se25 = se/10;
        b2 = [0:se25:-l1];
        b1 = [0:se25:l2];
        
        hs1 = my_conv(histc(df1, b1), 1);
        hs2 = my_conv(histc(-df2, b2), 1);
        

        mlow = min(max(hs1(1), hs1(2)), max(hs2(1), hs2(2)));
        plot(b1, hs1, 'Linewidth', 2)
        hold on
        plot(-b2, hs2, 'Linewidth', 2)
        hold off
        axis tight
         title({sprintf('%d vs %d', testID, simIDs(s)), sprintf('%2.2f and %2.2f', max(hs2)/mlow, max(hs1)/mlow)})
    end
end