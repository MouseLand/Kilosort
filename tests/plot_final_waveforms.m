%%
isV1pyr = rez.nbins(1:Nfilt)> 1000 & rez.ypos'>0 & rez.t2p(:,1)>10;
isV1pv  = rez.nbins(1:Nfilt)> 1000 & rez.ypos'>0 & rez.t2p(:,1)<=10;
W0 = alignW(W(:,:,1));

figure(1)
hist(rez.t2p(rez.nbins>1000,1), 1:1:nt0) 
xlabel('trough to peak of templates with >1000 spikes')
set(gcf, 'Color', 'w')
ylabel('number of templates')
% export_fig('fig1.pdf')

figure(2)
which_cells = find(isV1pyr);
[~, isort] = sort(rez.ypos(which_cells), 'ascend');
which_cells = which_cells(isort);
subplot(1,2,1)
imagesc(U(:, which_cells,1))
title('RS templates: spatial profile')
colormap('gray')
subplot(1,2,2)
plot(W0(:, which_cells,1))
axis tight
title('temporal profile')
set(gcf, 'Color', 'w')
% export_fig('fig2.pdf')

figure(3)
which_cells = find(isV1pv);
[~, isort] = sort(rez.ypos(which_cells), 'ascend');
which_cells = which_cells(isort);
subplot(1,2,1)
imagesc(U(:, which_cells,1))
colormap('gray')

title('FS templates: spatial profile')
subplot(1,2,2)
plot(W0(:, which_cells,1))
axis tight
title('temporal profile')
set(gcf, 'Color', 'w')

% export_fig('fig3.pdf')
%% bad  iNN 121, 478
iscell = rez.nbins(1:Nfilt)> 300;

[~, isortmu] = sort(mu, 'descend');

for i = 1:Nfilt
    
    ts = -.5:25:2000;
    iNN = isortmu(i);
   uu = histc(diff(st3pos(st3pos(:,2)==iNN, 1)), ts);
   subplot(2,2,1)
   bar([-fliplr(ts) ts(2:end)]', [flipud(uu); uu(2:end)])
   axis tight
   title(mu(iNN))
   
   subplot(2,2,2)
   hist(rez.st3pos(rez.st3pos(:,2)==iNN,3), 100)
   title(mu(iNN))
   axis tight
   
   [mWmax, iNN2] = max(mWtW(iNN, 1:Nfilt));
   uu = histc(diff(st3pos(st3pos(:,2)==iNN2, 1)), ts);
   subplot(2,2,4)
   bar([-fliplr(ts) ts(2:end)]', [flipud(uu); uu(2:end)])
   axis tight
   title(mu(iNN2))
   
   uu = histc(diff(st3pos(st3pos(:,2)==iNN2 | st3pos(:,2)==iNN, 1)), ts);
   subplot(2,2,3)
   bar([-fliplr(ts) ts(2:end)]', [flipud(uu); uu(2:end)])
    axis tight
    title(mWmax)
    
   pause
end

