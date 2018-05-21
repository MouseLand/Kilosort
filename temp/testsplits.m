
%% smooth the histogram
his2 = his;
his2(end, :) = 0;
his2 = my_conv2(his2,2, 1);

[hmax, imax] = max(his2, [], 1);

Nfilt = size(his2,2);
his2max = gpuArray.zeros(100, Nfilt, 2, 'single');

for j = 2:100
    his2max(j, :,1)  = max(his2max(j-1,:,1), his2(j, :));    
end
for j = 99:-1:1
    his2max(j, :,2)  = max(his2max(j+1,:,2), his2(j, :));    
end
his2max = min(his2max, [], 3);

dh = his2max - his2;

hmax = gpuArray.zeros(100, Nfilt, 'single');
hwid = gpuArray.zeros(1, Nfilt, 'single');
for j = 1:100
    hwid(dh(j,:)<1e-5) = 0;
    hwid =  hwid + dh(j,:);
    hmax(j, :) = hwid; 
end
hwid = gpuArray.zeros(1, Nfilt, 'single');
for j = 100:-1:1
    hmax(j, :) = hmax(j, :) + hwid; 
    hwid(dh(j,:)<1e-5) = 0;    
    hwid =  hwid + dh(j,:);    
end
%%

dharea = hmax./sum(his2,1);

%%

mh = max(dharea, [], 1);

[~, imax] = max((mh - dharea<1e-3) .* dh, [], 1);


%%
k =  ceil(rand * 100);

% k = 75;
figure(3)
plot(his2(:,k))
hold all
plot(his2max(:,k))

plot([imax(k) imax(k)], [his2(imax(k), k) his2max(imax(k), k)], '-k')
hold off


[sum(his(:,k)) mh(k)]
