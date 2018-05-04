function plot_waveforms(W)

W = W - repmat(mean(W,1), size(W,1), 1);

uu = max(abs(W(:)));

for i = 1:size(W,2)
   plot(i*uu/24 + W(:,i), 'k')
   hold all
end
hold off

axis tight