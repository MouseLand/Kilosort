uiopen('C:\Users\Marius\Downloads\sortBatches.fig',1)
h = gcf;
axesObjs = get(h, 'Children');  %axes handles
dataObjs = get(axesObjs, 'Children');
%%
cc = dataObjs{2}.CData;
%%
addpath('D:\Github\Kilosort2\preProcess')
cc = dataObjs{2}.CData;
[ccsort, isort] = sort_by_rastermap(cc);
%%
figure;
imagesc(ccsort, [-1 1])


