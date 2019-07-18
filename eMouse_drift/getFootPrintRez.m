function [unitPos, unitSize] = getFootPrintRez( rez )

%for testing standalone
% load('C:\Users\labadmin\Documents\emouse_drift\nodrift_sn_series\rand_pos_2X_KS2_release\rc03_rezFinal.mat')

%input rez from KS1 or KS2, return array of unit "sizes"
%sites included in unit defined as those with abs(1st component of U) > 
% 0.1*max(abs(U(:,1))
% size = diagonal of rectangle that contains the center of the sites, i.e.
% sqrt( (maxY-minY)^2 + (maxX-minX)^2 );

U = rez.U;
[nChan, nClu, nPC] = size(U);
unitPos = zeros(nClu,1);
unitSize = zeros(nClu,1);

nDist = 30;     %neighbor distance, in um
incFrac = 0.1;    %included sites have projection onto first PC this fraction of max


for j = 1:nClu
    currU = U(:,j,1);
    [maxU, maxI] = max(currU);
    [minU, minI] = min(currU);
    maxP = maxU - minU;
    unitPos(j) = rez.yc(maxI);
    inclSite = find(abs(currU) > incFrac*maxP);
    inclDist=[];
    for k = 1:numel(inclSite)
        inclDist(k) = sqrt( (rez.yc(inclSite(k)) - rez.yc(maxI))^2 + ...
                       (rez.xc(inclSite(k)) - rez.xc(maxI))^2 );
    end
    [sortDist, ~] = sort(inclDist);
    
    nextNearest = 0;
    currI = 1;
    while ((nextNearest < nDist) && currI < numel(inclDist))
        currI = currI + 1;
        nextNearest = sortDist(currI)-sortDist(currI-1);      
    end
    if( nextNearest > nDist )
        %then last site tested was not contiguous with other included sites
        currI = currI - 1;
    end
    %fprintf( 'cluIndex, numIncl, currI: %d, %d, %d\n', j, numel(inclSite),currI);
    %unit radius est = sortDist(currI)
    unitSize(j) = 2*sortDist(currI);
                   
end


end