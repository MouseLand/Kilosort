function [nrshifts,F0] = align_nonrigid(F,ishift,iblk,sig)
%ALIGN_NONRIGID Diffeomorphic Demons non-rigid image registration
%   Detailed explanation goes here

if nargin==3
    % -10:10 pixels = -50:50 um sptial gaussian(5um spaceing), -3:3 pixels = -6:6 sec temporal gaussian(~2sec batch)
    sig=[5,1.5];
end

nbatch = size(F,3);
ndepth = size(F,1);
brshifts = interp1(iblk, ishift', 1:ndepth, 'makima', 'extrap'); % block rigid shifts on frame depths

% spike amplitude image
I = zeros(ndepth,nbatch);
for i=1:nbatch
    for j=1:ndepth
        I(j,i)= sum(F(j,:,i).*(1:size(F,2)))/sum(F(j,:,i)); % mean amplitude
    end
end
bI = imwarp(I,cat(3,zeros(size(brshifts)),-brshifts));
plotdrift(flipud(-brshifts),flipud(I),flipud(bI),'Block-Rigid Registration');

% non-rigid image registration
[df,F0] = nonrigid(F);
nrshifts = cellfun(@(x)median(x(:,:,2),2),df,'UniformOutput',false); % displacement field on depths
nrshifts = [nrshifts{:}];

% high freqency shifts on both spatial and temporal are less likely true,
% and more likely reflect noise in spike amplitude distribution: F
if ~isempty(sig)
    nrshifts = imgaussfilt(nrshifts,sig,'FilterDomain','frequency');
end
dI = imwarp(I,cat(3,zeros(size(nrshifts)),nrshifts));
plotdrift(flipud(nrshifts),flipud(I),flipud(dI),'Non-Rigid Registration');
nrshifts=-nrshifts';

    function plotdrift(d,im,dcim,titlestr)
        if nargin==3
            titlestr='';
        end
        
        figure('units','normalized','position',[0.15 0.4 0.7 0.35])
        t=tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
        
        nexttile
        imagesc(d)
        title('Estimated drift')
        set(gca,'yticklabel',[])
        box off
        
        nexttile
        imagesc(im)
        title('Drift map')
        set(gca,'yticklabel',[])
        box off
        
        nexttile
        imagesc(dcim)
        title('Estimated corrected drift map')
        set(gca,'yticklabel',[])
        box off
        
        title(t,titlestr)
        xlabel(t,'batch number')
        ylabel(t,'depth')
        drawnow
    end

end

