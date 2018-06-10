function make_update(ibatch, niter, nsp, mu, W, U, nst)

Nfilt = size(W,2);

fprintf('%3.0f sec, %d / %d batches, %d units, nspks: %2.2f, mu: %2.2f, nst0: %2.2f \n', ...
    toc, ibatch, niter, Nfilt, median(nsp), median(mu), nst)

figure(2)
subplot(2,2,1)
imagesc(W(:,:,1))

subplot(2,2,2)
imagesc(U(:,:,1))

subplot(2,2,3)
plot(mu)
ylim([0 100])
xlim([0 size(W,2)+1])

subplot(2,2,4)
semilogx(1+nsp, mu, '.')
xlim([0 200])
ylim([0 100])

drawnow