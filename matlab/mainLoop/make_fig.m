function make_fig(W, U, mu, nsp)

subplot(2,2,1)
imagesc(W(:,:,1))
title('Temporal Components')
xlabel('Unit number');
ylabel('Time (samples)');

subplot(2,2,2)
imagesc(U(:,:,1))
title('Spatial Components')
xlabel('Unit number');
ylabel('Channel number');

subplot(2,2,3)
plot(mu)
ylim([0 100])
title('Unit Amplitudes')
xlabel('Unit number');
ylabel('Amplitude (arb. units)');

subplot(2,2,4)
semilogx(1+nsp, mu, '.')
ylim([0 100])
xlim([0 100])
title('Amplitude vs. Spike Count')
xlabel('Spike Count');
ylabel('Amplitude (arb. units)');
drawnow