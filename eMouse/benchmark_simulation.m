function benchmark_simulation(rez, GTfilepath)

load(GTfilepath)

try
    testClu = 1 + rez.st3(:,5) ; % if the auto merges were performed
    flag = 1;
catch
    testClu = rez.st3(:,2) ;% no attempt to merge clusters
    flag = 0;
end

testRes = rez.st3(:,1) ;

[allScores, allFPrates, allMissRates, allMerges] = ...
    compareClustering2(gtClu, gtRes, testClu, testRes, []);

%
clid = unique(gtClu);
clear gtimes
for k = 1:length(clid)
    gtimes{k} = double(gtRes(gtClu==clid(k)));
end
%%

figure

plot(sort(cellfun(@(x) x(1), allFPrates)), '-*b', 'Linewidth', 2)
hold all
plot(sort(cellfun(@(x) x(1), allMissRates)), '-*r', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allFPrates)), 'b', 'Linewidth', 2)
plot(sort(cellfun(@(x) x(end), allMissRates)), 'r', 'Linewidth', 2)
ylim([0 1])
box off

finalScores = cellfun(@(x) x(end), allScores);
fprintf('%d / %d good cells, score > 0.8 (pre-merge) \n', sum(cellfun(@(x) x(1), allScores)>.8), numel(allScores))
fprintf('%d / %d good cells, score > 0.8 (post-merge) \n', sum(cellfun(@(x) x(end), allScores)>.8), numel(allScores))

nMerges = cellfun(@(x) numel(x)-1, allMerges);
fprintf('Mean merges per good cell %2.2f \n', mean(nMerges(finalScores>.8)))

% disp(cellfun(@(x) x(end), allScores))

xlabel('ground truth cluster')
ylabel('fractional error')

legend('false positives (initial)', 'miss rates (initial)', 'false positives (best)', 'miss rates (best)')
legend boxoff
set(gca, 'Fontsize', 20)
set(gcf, 'Color', 'w')

if flag==1
   title('After Kilosort AUTO merges') 
else
    title('Before Kilosort AUTO merges')
end
