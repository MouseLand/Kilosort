function rez = merge_posthoc2(rez)
%fracse = 0.1;
mu = rez.mu;
fracse = rez.ops.fracse;

ops = rez.ops;
LAM = ops.lam(3) * (20./mu).^2;
Nfilt = rez.ops.Nfilt;

Cmerge = Inf *ones(Nfilt);
tfi = rez.iNeigh;
tf = rez.cProj;
clusterIDs = rez.st3(:,2);
%
nSpikes = size(rez.st3,1);

fmax = zeros(nSpikes,1, 'single');
pairs = {};
for testID = 1:Nfilt
    spikesTest = clusterIDs==testID;
%     tfnew = bsxfun(@plus, tf(spikesTest, :), LAM(tfi(:, testID))'.*mu(tfi(:, testID))');
%     tf(spikesTest, :) = bsxfun(@rdivide, tfnew, sqrt(1+LAM(tfi(:, testID)))');
    
    pp = tfi(:, testID);
    pp(pp==testID) = [];
    pairs{testID} = pp;
    [~, isame] = min( abs(tfi(:, testID)-testID));
    fmax(spikesTest, 1) = tf(spikesTest, isame);
end


%
inewclust = 0;
clear iMegaC
picked = zeros(Nfilt, 1);
% tic
while 1    
    [maxseed, iseed] = max(rez.nbins(1:Nfilt) .* (1-picked), [], 1);
%     [maxseed, iseed] = max(mu(1:Nfilt) .* (1-picked), [], 1);
    if maxseed<500
        break;
    end
    picked(iseed) = 1;
    % 21, 69,
    %
%     iseed = 410;
    
    run_list = [iseed];
    pair_list = pairs{iseed};
    strun = find(clusterIDs==iseed);
    
    
    while ~isempty(pair_list)
        %
%         picked_pairs = rez.nbins(pair_list);
        
        [mmax, ipair] = max(rez.nbins(pair_list));
        
        
        if mmax<100
            break;
        end
        
        ipair = pair_list(ipair);
        
        %
        imm = ismember(tfi(:, ipair), run_list);
        if sum(imm)
            %
            new_spikes = find(clusterIDs==ipair);
            f1new = max(tf(new_spikes, imm), [], 2);
            
            f2new = fmax(new_spikes);
            
            f1old = fmax(strun);
            f2old = NaN * ones(numel(f1old), 1, 'single');
            i0 = 0;
            for j = 1:length(run_list)
                ifeat = find(tfi(:, run_list(j))==ipair);
                if ~isempty(ifeat)
                    f2old(i0 + (1:rez.nbins(run_list(j))),1) = ...
                        tf(clusterIDs==run_list(j), ifeat);
                    i0 = i0 + rez.nbins(run_list(j));
                end
            end
            
            f1old(isnan(f2old))=[];
            f2old(isnan(f2old))=[];
            mo = merging_score(f1old - f2old, f1new-f2new, ops.fracse);
            
            
            if mo<3
                strun = cat(1, strun, new_spikes);
                run_list(end+1) = ipair;
                picked(ipair)   = 1;
                if mmax>300
                    pair_list = unique(cat(1, pair_list, pairs{ipair}));
                    pair_list(ismember(pair_list, run_list)) = [];
                end                
            end
        end
        pair_list(pair_list==ipair) = [];
    end
    
    inewclust = inewclust + 1;
    
    iMegaC{inewclust} = run_list;
%     [sum(picked) run_list]
end

% toc
%

iMega = zeros(Nfilt, 1);
for i = 1:length(iMegaC)
   iMega(iMegaC{i}) = iMegaC{i}(1); 
end
rez.iMega = iMega;
rez.iMegaC = iMegaC;


rez.st3(:,5) = iMega(rez.st3(:,2));